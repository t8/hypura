#include "hypura_buft.h"
#include "ggml-backend-impl.h"
#include <stdlib.h>
#include <string.h>

/* ---------- Context structs ---------- */

typedef struct {
    hypura_on_tensor_loaded_t on_tensor_loaded;
    hypura_on_tensor_init_t   on_tensor_init;
    void *rust_ctx;
} hypura_buft_context;

typedef struct {
    void  *base;
    size_t size;
    hypura_buft_context *buft_ctx;
} hypura_buffer_context;

/* ---------- Buffer vtable ---------- */

static void hypura_buf_free(ggml_backend_buffer_t buffer) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    if (ctx) {
        free(ctx->base);
        free(ctx);
    }
}

static void *hypura_buf_get_base(ggml_backend_buffer_t buffer) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    return ctx->base;
}

static enum ggml_status hypura_buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    if (ctx->buft_ctx->on_tensor_init) {
        ctx->buft_ctx->on_tensor_init(ctx->buft_ctx->rust_ctx, tensor, tensor->name);
    }
    return GGML_STATUS_SUCCESS;
}

static void hypura_buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor,
                                     uint8_t value, size_t offset, size_t size) {
    (void)buffer;
    memset((uint8_t *)tensor->data + offset, value, size);
}

static void hypura_buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor,
                                  const void *data, size_t offset, size_t size) {
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;

    /* Copy tensor data into our buffer */
    memcpy((uint8_t *)tensor->data + offset, data, size);

    /* Notify Rust of this tensor's location */
    if (ctx->buft_ctx->on_tensor_loaded && offset == 0) {
        size_t buf_offset = (uint8_t *)tensor->data - (uint8_t *)ctx->base;
        ctx->buft_ctx->on_tensor_loaded(ctx->buft_ctx->rust_ctx, tensor->name, buf_offset, size, ctx->base);
    }
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
    hypura_buffer_context *ctx = (hypura_buffer_context *)buffer->context;
    memset(ctx->base, value, ctx->size);
}

/* ---------- Buffer type vtable ---------- */

static const char *hypura_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "Hypura_NVMe";
}

static ggml_backend_buffer_t hypura_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    /* Page-align the allocation for direct I/O compatibility */
    size_t aligned_size = (size + 4095) & ~(size_t)4095;

    void *base = NULL;
    int ret = posix_memalign(&base, 4096, aligned_size);
    if (ret != 0 || base == NULL) {
        return NULL;
    }

    hypura_buffer_context *buf_ctx = (hypura_buffer_context *)calloc(1, sizeof(hypura_buffer_context));
    if (!buf_ctx) {
        free(base);
        return NULL;
    }
    buf_ctx->base     = base;
    buf_ctx->size     = aligned_size;
    buf_ctx->buft_ctx = (hypura_buft_context *)buft->context;

    struct ggml_backend_buffer_i iface = {
        .free_buffer  = hypura_buf_free,
        .get_base     = hypura_buf_get_base,
        .init_tensor  = hypura_buf_init_tensor,
        .memset_tensor = hypura_buf_memset_tensor,
        .set_tensor   = hypura_buf_set_tensor,
        .get_tensor   = hypura_buf_get_tensor,
        .cpy_tensor   = hypura_buf_cpy_tensor,
        .clear        = hypura_buf_clear,
        .reset        = NULL,
    };

    return ggml_backend_buffer_init(buft, iface, buf_ctx, aligned_size);
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

/* ---------- Public API ---------- */

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
