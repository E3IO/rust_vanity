pub struct VanityConfig {
    pub prefixes: Vec<String>,
    pub suffixes: Vec<String>,
    pub ignore_case: bool,
    pub gpu_enabled: bool,
    pub threads: usize,
    pub batch_size: usize,
    pub output_file: Option<String>,
}

impl Default for VanityConfig {
    fn default() -> Self {
        Self {
            prefixes: vec!["hello".to_string()],
            suffixes: vec![],
            ignore_case: true,
            gpu_enabled: true,
            threads: 8,
            batch_size: 100000,
            output_file: Some("found_keys.txt".to_string()),
        }
    }
}

impl VanityConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefixes.push(prefix.to_string());
        self
    }

    pub fn with_suffix(mut self, suffix: &str) -> Self {
        self.suffixes.push(suffix.to_string());
        self
    }

    pub fn with_prefixes(mut self, prefixes: Vec<String>) -> Self {
        self.prefixes = prefixes;
        self
    }

    pub fn with_suffixes(mut self, suffixes: Vec<String>) -> Self {
        self.suffixes = suffixes;
        self
    }

    pub fn with_ignore_case(mut self, ignore_case: bool) -> Self {
        self.ignore_case = ignore_case;
        self
    }

    pub fn with_gpu_enabled(mut self, gpu_enabled: bool) -> Self {
        self.gpu_enabled = gpu_enabled;
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_output_file(mut self, output_file: Option<String>) -> Self {
        self.output_file = output_file;
        self
    }
} 