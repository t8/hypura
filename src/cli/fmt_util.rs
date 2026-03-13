pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{bytes} B")
    }
}

pub fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}

pub fn format_bandwidth(bytes_per_sec: u64) -> String {
    let gb_s = bytes_per_sec as f64 / 1e9;
    if gb_s >= 1.0 {
        format!("{gb_s:.1} GB/s")
    } else {
        format!("{:.0} MB/s", bytes_per_sec as f64 / 1e6)
    }
}
