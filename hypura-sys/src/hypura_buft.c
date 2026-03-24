#include "hypura_buft.h"
#include "ggml-backend-impl.h"
#include <stdlib.h>
#include <string.h>

/* ── Platform-specific anonymous memory ──────────────────────────────────── */

#ifdef _WIN32
#  include <windows.h>

static void *platform_alloc_pages(size_t size) {
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

static void platform_free_pages(void *addr, size_t size) {
    (void)size;
    if (addr) VirtualFree(addr, 0, MEM_RELEASE);
}

#else
#  include <sys/mman.h>

static void *platform_alloc_pages(size_t size) {
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_ANON | MAP_PRIVATE, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

static void platform_free_pages(void *addr, size_t size) {
    if (addr) munmap(addr, size);
}
#endif

/* ── Context structs ─────────────────────────────────────────────────────── */

typedef struct {
    hypura_on_tensor_loaded_t on_tensor_loaded;
    hypura_on_tensor_init_t   on_tensor_init;
    void *rust_ctx;
    ggml_backend_buffer_t last_buffer;  /* most recently allocated buffer */
} hypura_buft_context;

typedef struct {
    void  *base;       /* anonymous page buffer (loading phase) */
    size_t size;
    hypura_buft_context *buft_ctx;
    void  *pool_base;  /* pool buffer (inference phase, expert-streaming) */
    size_t pool_size;
    int    pool_active; /* 0 = loading phase, 1 = pool phase */
} hypura_buffer_context;

/* ── Buffer vtable ───────────────────────────────────────────────────────── */

static void hypura_buf_free(ggml_backend_buffer_t buffer) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    if (ctx) {
        if (ctx->base && ctx->size > 0) {
            platform_free_pages(ctx->base, ctx->size);
        }
        if (ctx->pool_base && ctx->pool_size > 0) {
            platform_free_pages(ctx->pool_base, ctx->pool_size);
        }
        free(ctx);
    }
}

static void *hypura_buf_get_base(ggml_backend_buffer_t buffer) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    return ctx->base;
}

static enum ggml_status hypura_buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;

    /* Register tensor location here rather than set_tensor, because llama.cpp
     * may bypass set_tensor for host buffers when use_mmap=false (reads directly
     * into tensor->data via fread). init_tensor is ALWAYS called. */
    if (ctx->buft_ctx->on_tensor_loaded) {
        size_t buf_offset = (uint8_t *)tensor->data - (uint8_t *)ctx->base;
        size_t size = ggml_nbytes(tensor);
        ctx->buft_ctx->on_tensor_loaded(ctx->buft_ctx->rust_ctx, tensor->name,
                                        buf_offset, size, ctx->base);
    }

    if (ctx->buft_ctx->on_tensor_init) {
        ctx->buft_ctx->on_tensor_init(ctx->buft_ctx->rust_ctx, tensor, tensor->name);
    }
    return GGML_STATUS_SUCCESS;
}

static void hypura_buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor,
                                     uint8_t value, size_t offset, size_t size) {
    /* Skip — data loaded lazily from file via pread during inference */
    (void)buffer; (void)tensor; (void)value; (void)offset; (void)size;
}

static void hypura_buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor,
                                  const void *data, size_t offset, size_t size) {
    /* Skip memcpy — data loaded lazily from file via pread during inference.
     * Tensor registration happens in init_tensor instead. */
    (void)buffer; (void)tensor; (void)data; (void)offset; (void)size;
}

static void hypura_buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor *tensor,
                                  void *data, size_t offset, size_t size) {
    (void)buffer;
    memcpy(data, (const uint8_t *)tensor->data + offset, size);
}

static bool hypura_buf_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor *src,
                                  struct ggml_tensor *dst) {
    (void)buffer;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void hypura_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    /* Skip — data loaded lazily from file via pread during inference */
    (void)buffer; (void)value;
}

/* ── Buffer type vtable ──────────────────────────────────────────────────── */

static const char *hypura_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "Hypura_NVMe";
}

static ggml_backend_buffer_t hypura_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    /* Page-align the allocation for direct I/O compatibility.
     * Use platform_alloc_pages (mmap/VirtualAlloc): pages are lazily committed
     * on first access, so a large virtual reservation doesn't immediately
     * consume physical memory or GPU working set. */
    size_t aligned_size = (size + 4095) & ~(size_t)4095;

    void *base = platform_alloc_pages(aligned_size);
    if (!base) {
        return NULL;
    }

    hypura_buffer_context *buf_ctx = (hypura_buffer_context *)calloc(1, sizeof(hypura_buffer_context));
    if (!buf_ctx) {
        platform_free_pages(base, aligned_size);
        return NULL;
    }
    buf_ctx->base     = base;
    buf_ctx->size     = aligned_size;
    buf_ctx->buft_ctx = (hypura_buft_context *)buft->context;

    struct ggml_backend_buffer_i iface = {
        .free_buffer   = hypura_buf_free,
        .get_base      = hypura_buf_get_base,
        .init_tensor   = hypura_buf_init_tensor,
        .memset_tensor = hypura_buf_memset_tensor,
        .set_tensor    = hypura_buf_set_tensor,
        .get_tensor    = hypura_buf_get_tensor,
        .cpy_tensor    = hypura_buf_cpy_tensor,
        .clear         = hypura_buf_clear,
        .reset         = NULL,
    };

    ggml_backend_buffer_t buf = ggml_backend_buffer_init(buft, iface, buf_ctx, aligned_size);

    /* Store buffer handle for Rust-side pool activation */
    hypura_buft_context *buft_ctx = (hypura_buft_context *)buft->context;
    buft_ctx->last_buffer = buf;

    return buf;
}

static size_t hypura_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 32; /* matches TENSOR_ALIGNMENT in ggml */
}

static size_t hypura_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return (size_t)-1; /* no limit */
}

static bool hypura_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true; /* critical: CPU backend requires is_host=true */
}

/* ── Public API ──────────────────────────────────────────────────────────── */

ggml_backend_buffer_type_t hypura_buft_create(
    hypura_on_tensor_loaded_t on_tensor_loaded,
    hypura_on_tensor_init_t on_tensor_init,
    void *rust_ctx
) {
    hypura_buft_context *ctx = (hypura_buft_context *)calloc(1, sizeof(hypura_buft_context));
    if (!ctx) return NULL;
    ctx->on_tensor_loaded = on_tensor_loaded;
    ctx->on_tensor_init   = on_tensor_init;
    ctx->rust_ctx         = rust_ctx;

    ggml_backend_buffer_type_t buft =
        (ggml_backend_buffer_type_t)calloc(1, sizeof(struct ggml_backend_buffer_type));
    if (!buft) {
        free(ctx);
        return NULL;
    }

    buft->iface.get_name       = hypura_buft_get_name;
    buft->iface.alloc_buffer   = hypura_buft_alloc_buffer;
    buft->iface.get_alignment  = hypura_buft_get_alignment;
    buft->iface.get_max_size   = hypura_buft_get_max_size;
    buft->iface.get_alloc_size = NULL; /* use default ggml_nbytes */
    buft->iface.is_host        = hypura_buft_is_host;
    buft->device  = NULL;
    buft->context = ctx;

    return buft;
}

void hypura_buft_free(ggml_backend_buffer_type_t buft) {
    if (buft) {
        free(buft->context);
        free(buft);
    }
}

void *hypura_buffer_get_base_ptr(ggml_backend_buffer_t buffer) {
    if (!buffer) return NULL;
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    return ctx ? ctx->base : NULL;
}

/* ── Pool buffer API ─────────────────────────────────────────────────────── */

int hypura_buffer_init_pool(ggml_backend_buffer_t buffer, size_t pool_size) {
    if (!buffer) return -1;
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    if (!ctx) return -1;

    size_t aligned = (pool_size + 4095) & ~(size_t)4095;
    void *pool = platform_alloc_pages(aligned);
    if (!pool) return -1;

    ctx->pool_base = pool;
    ctx->pool_size = aligned;
    return 0;
}

void hypura_buffer_release_loading_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer) return;
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    if (!ctx) return;

    if (ctx->base && ctx->size > 0) {
        platform_free_pages(ctx->base, ctx->size);
        ctx->base = NULL;
        ctx->size = 0;
    }
    ctx->pool_active = 1;
}

void *hypura_buffer_get_pool_base(ggml_backend_buffer_t buffer) {
    if (!buffer) return NULL;
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    return ctx ? ctx->pool_base : NULL;
}

size_t hypura_buffer_get_pool_size(ggml_backend_buffer_t buffer) {
    if (!buffer) return 0;
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    return ctx ? ctx->pool_size : 0;
}

ggml_backend_buffer_t hypura_buft_get_last_buffer(ggml_backend_buffer_type_t buft) {
    if (!buft) return NULL;
    hypura_buft_context *ctx = (hypura_buft_context *)buft->context;
    return ctx ? ctx->last_buffer : NULL;
}
