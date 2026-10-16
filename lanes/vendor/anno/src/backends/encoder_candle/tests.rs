use super::*;

#[test]
fn test_encoder_config_defaults() {
    let bert = EncoderConfig::bert_base();
    assert_eq!(bert.hidden_size, 768);
    assert_eq!(bert.max_position_embeddings, 512);
    assert!(!bert.use_rope);

    let modern = EncoderConfig::modernbert_base();
    assert_eq!(modern.hidden_size, 768);
    assert_eq!(modern.max_position_embeddings, 8192);
    assert!(modern.use_rope);
    assert!(modern.use_geglu);
}

#[test]
fn test_modernbert_large() {
    let config = EncoderConfig::modernbert_large();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_hidden_layers, 28);
}

#[test]
fn test_deberta_v3_base_config() {
    let config = EncoderConfig::deberta_v3_base();
    assert_eq!(config.vocab_size, 128100);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.max_position_embeddings, 512);
    assert!(!config.use_rope);
    assert!(!config.use_geglu);
    assert!(config.use_pre_norm);
    assert!((config.layer_norm_eps - 1e-7).abs() < 1e-15);
}

#[test]
fn test_deberta_v3_large_config() {
    let config = EncoderConfig::deberta_v3_large();
    assert_eq!(config.vocab_size, 128100);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.intermediate_size, 4096);
    assert_eq!(config.max_position_embeddings, 512);
    assert!(!config.use_rope);
    assert!(!config.use_geglu);
    assert!(config.use_pre_norm);
    // Shared vocab and eps with base
    assert_eq!(
        config.vocab_size,
        EncoderConfig::deberta_v3_base().vocab_size
    );
    assert!(
        (config.layer_norm_eps - EncoderConfig::deberta_v3_base().layer_norm_eps).abs() < 1e-15
    );
}

#[test]
fn test_modernbert_base_config() {
    let config = EncoderConfig::modernbert_base();
    assert_eq!(config.vocab_size, 50368);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 22);
    assert_eq!(config.intermediate_size, 1152);
    assert_eq!(config.max_position_embeddings, 8192);
    assert!(config.use_rope);
    assert!(config.use_geglu);
    assert!(config.use_pre_norm);
    assert_eq!(config.hidden_dropout_prob, 0.0);
    assert!((config.rope_theta - 160000.0).abs() < f64::EPSILON);
}

#[test]
fn test_from_model_name_dispatch() {
    // ModernBERT variants
    let cfg = EncoderConfig::from_model_name("answerdotai/ModernBERT-base");
    assert_eq!(cfg.hidden_size, 768);
    assert!(cfg.use_rope);
    assert_eq!(cfg.num_hidden_layers, 22);

    let cfg = EncoderConfig::from_model_name("answerdotai/ModernBERT-large");
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 28);

    // DeBERTa variants
    let cfg = EncoderConfig::from_model_name("microsoft/deberta-v3-base");
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.vocab_size, 128100);
    assert!(!cfg.use_rope);

    let cfg = EncoderConfig::from_model_name("microsoft/deberta-v3-large");
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);

    // Unknown model falls back to BERT-base
    let cfg = EncoderConfig::from_model_name("some-unknown/model-name");
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.vocab_size, 30522);
    assert!(!cfg.use_rope);
    assert!(!cfg.use_geglu);
}

#[test]
fn test_from_model_name_case_insensitive() {
    // Dispatch should be case-insensitive
    let lower = EncoderConfig::from_model_name("modernbert-base");
    let upper = EncoderConfig::from_model_name("MODERNBERT-BASE");
    assert_eq!(lower.hidden_size, upper.hidden_size);
    assert_eq!(lower.num_hidden_layers, upper.num_hidden_layers);

    let lower = EncoderConfig::from_model_name("deberta-large");
    let upper = EncoderConfig::from_model_name("DEBERTA-LARGE");
    assert_eq!(lower.hidden_size, upper.hidden_size);
    assert_eq!(lower.num_hidden_layers, upper.num_hidden_layers);
}

#[test]
fn test_architecture_properties() {
    // RoPE: only ModernBERT
    assert!(EncoderArchitecture::ModernBert.uses_rope());
    assert!(!EncoderArchitecture::Bert.uses_rope());
    assert!(!EncoderArchitecture::DeBertaV3.uses_rope());

    // Max length: 512 for BERT/DeBERTa, 8192 for ModernBERT
    assert_eq!(EncoderArchitecture::Bert.max_length(), 512);
    assert_eq!(EncoderArchitecture::DeBertaV3.max_length(), 512);
    assert_eq!(EncoderArchitecture::ModernBert.max_length(), 8192);

    // Display strings
    assert_eq!(EncoderArchitecture::Bert.as_str(), "BERT");
    assert_eq!(EncoderArchitecture::DeBertaV3.as_str(), "DeBERTa-v3");
    assert_eq!(EncoderArchitecture::ModernBert.as_str(), "ModernBERT");
}

#[test]
fn test_architecture_default_config_consistency() {
    // default_config() should return the same config as the named constructor
    let arch_cfg = EncoderArchitecture::Bert.default_config();
    let direct_cfg = EncoderConfig::bert_base();
    assert_eq!(arch_cfg.hidden_size, direct_cfg.hidden_size);
    assert_eq!(arch_cfg.num_hidden_layers, direct_cfg.num_hidden_layers);
    assert_eq!(arch_cfg.vocab_size, direct_cfg.vocab_size);

    let arch_cfg = EncoderArchitecture::ModernBert.default_config();
    let direct_cfg = EncoderConfig::modernbert_base();
    assert_eq!(arch_cfg.hidden_size, direct_cfg.hidden_size);
    assert_eq!(arch_cfg.num_hidden_layers, direct_cfg.num_hidden_layers);
    assert_eq!(arch_cfg.use_rope, direct_cfg.use_rope);

    let arch_cfg = EncoderArchitecture::DeBertaV3.default_config();
    let direct_cfg = EncoderConfig::deberta_v3_base();
    assert_eq!(arch_cfg.hidden_size, direct_cfg.hidden_size);
    assert_eq!(arch_cfg.num_hidden_layers, direct_cfg.num_hidden_layers);
}

#[test]
fn test_architecture_default_is_modernbert() {
    let arch = EncoderArchitecture::default();
    assert_eq!(arch, EncoderArchitecture::ModernBert);
}

#[test]
fn test_encoder_config_default_is_bert_base() {
    let cfg = EncoderConfig::default();
    let bert = EncoderConfig::bert_base();
    assert_eq!(cfg.hidden_size, bert.hidden_size);
    assert_eq!(cfg.vocab_size, bert.vocab_size);
    assert_eq!(cfg.num_hidden_layers, bert.num_hidden_layers);
}

#[cfg(feature = "candle")]
#[test]
fn test_geglu() {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1., (2, 8), &device).unwrap();
    let result = super::implementations::candle_impl::geglu(&x);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.dims(), &[2, 4]);
}

#[cfg(feature = "candle")]
#[test]
fn test_geglu_various_sizes() {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;

    // GeGLU always halves the last dimension
    for dim in [4, 16, 64, 256] {
        let x = Tensor::randn(0f32, 1., (1, dim), &device).unwrap();
        let result = super::implementations::candle_impl::geglu(&x).unwrap();
        assert_eq!(
            result.dims(),
            &[1, dim / 2],
            "GeGLU should halve dim={}",
            dim
        );
    }

    // Batch dimension preserved
    let x = Tensor::randn(0f32, 1., (5, 32), &device).unwrap();
    let result = super::implementations::candle_impl::geglu(&x).unwrap();
    assert_eq!(result.dims(), &[5, 16]);
}

#[cfg(feature = "candle")]
#[test]
fn test_rope_cache_shape() {
    use super::implementations::candle_impl::RotaryEmbedding;
    use candle_core::Device;

    let head_dim = 64;
    let max_seq_len = 128;
    let theta = 10000.0;
    let device = Device::Cpu;

    let rope = RotaryEmbedding::new(head_dim, max_seq_len, theta, &device).unwrap();

    // Apply to a tensor: [batch=1, seq=16, heads=1, head_dim=64]
    // (RoPE broadcasts cos/sin over the head dimension via unsqueeze)
    let x = candle_core::Tensor::randn(0f32, 1., (1, 16, 1, head_dim), &device).unwrap();
    let result = rope.apply(&x, 0).unwrap();

    // Output shape must match input shape exactly
    assert_eq!(result.dims(), &[1, 16, 1, head_dim]);

    // Verify partial-sequence apply (start_pos > 0)
    let x_short = candle_core::Tensor::randn(0f32, 1., (1, 8, 1, head_dim), &device).unwrap();
    let result = rope.apply(&x_short, 10).unwrap();
    assert_eq!(result.dims(), &[1, 8, 1, head_dim]);
}

#[cfg(feature = "candle")]
#[test]
fn test_best_device_returns_ok() {
    // best_device() should always succeed (falls back to CPU)
    let device = super::implementations::candle_impl::best_device();
    assert!(device.is_ok());
}

// =========================================================================
// Architecture: default_model_id dispatch
// =========================================================================

#[test]
fn test_architecture_default_model_ids() {
    assert_eq!(
        EncoderArchitecture::Bert.default_model_id(),
        "google-bert/bert-base-uncased"
    );
    assert_eq!(
        EncoderArchitecture::DeBertaV3.default_model_id(),
        "microsoft/deberta-v3-base"
    );
    assert_eq!(
        EncoderArchitecture::ModernBert.default_model_id(),
        "answerdotai/ModernBERT-base"
    );
}

// =========================================================================
// Architecture: Display trait
// =========================================================================

#[test]
fn test_architecture_display() {
    assert_eq!(format!("{}", EncoderArchitecture::Bert), "BERT");
    assert_eq!(format!("{}", EncoderArchitecture::DeBertaV3), "DeBERTa-v3");
    assert_eq!(format!("{}", EncoderArchitecture::ModernBert), "ModernBERT");
}

// =========================================================================
// Config: head dimension divisibility invariant
// =========================================================================

#[test]
fn test_config_head_dim_divisibility() {
    // hidden_size must be evenly divisible by num_attention_heads for all configs
    let configs = [
        ("bert_base", EncoderConfig::bert_base()),
        ("modernbert_base", EncoderConfig::modernbert_base()),
        ("modernbert_large", EncoderConfig::modernbert_large()),
        ("deberta_v3_base", EncoderConfig::deberta_v3_base()),
        ("deberta_v3_large", EncoderConfig::deberta_v3_large()),
    ];
    for (name, cfg) in &configs {
        assert_eq!(
            cfg.hidden_size % cfg.num_attention_heads,
            0,
            "{}: hidden_size {} not divisible by num_attention_heads {}",
            name,
            cfg.hidden_size,
            cfg.num_attention_heads
        );
    }
}

// =========================================================================
// Config: bert_base detailed field validation
// =========================================================================

#[test]
fn test_bert_base_full_config() {
    let cfg = EncoderConfig::bert_base();
    assert_eq!(cfg.vocab_size, 30522);
    assert_eq!(cfg.hidden_size, 768);
    assert_eq!(cfg.num_attention_heads, 12);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert_eq!(cfg.intermediate_size, 3072);
    assert_eq!(cfg.max_position_embeddings, 512);
    assert_eq!(cfg.hidden_dropout_prob, 0.1);
    assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-20);
    assert!(!cfg.use_rope);
    assert!(!cfg.use_geglu);
    assert!((cfg.rope_theta - 10000.0).abs() < f64::EPSILON);
    assert!(!cfg.use_pre_norm); // classic BERT uses post-norm
}

// =========================================================================
// Config: large > base ordering invariant
// =========================================================================

#[test]
fn test_large_configs_wider_than_base() {
    // ModernBERT: large has wider hidden and more layers than base
    let mb_base = EncoderConfig::modernbert_base();
    let mb_large = EncoderConfig::modernbert_large();
    assert!(mb_large.hidden_size > mb_base.hidden_size);
    assert!(mb_large.num_hidden_layers > mb_base.num_hidden_layers);
    assert!(mb_large.num_attention_heads > mb_base.num_attention_heads);
    assert!(mb_large.intermediate_size > mb_base.intermediate_size);
    // Same vocab and context window
    assert_eq!(mb_large.vocab_size, mb_base.vocab_size);
    assert_eq!(
        mb_large.max_position_embeddings,
        mb_base.max_position_embeddings
    );

    // DeBERTa-v3: large has wider hidden and more layers than base
    let db_base = EncoderConfig::deberta_v3_base();
    let db_large = EncoderConfig::deberta_v3_large();
    assert!(db_large.hidden_size > db_base.hidden_size);
    assert!(db_large.num_hidden_layers > db_base.num_hidden_layers);
    assert!(db_large.num_attention_heads > db_base.num_attention_heads);
    assert!(db_large.intermediate_size > db_base.intermediate_size);
    // Same vocab
    assert_eq!(db_large.vocab_size, db_base.vocab_size);
}

// =========================================================================
// from_model_name: HuggingFace-style paths with org prefix
// =========================================================================

#[test]
fn test_from_model_name_with_org_prefixes() {
    // Full HuggingFace model paths should dispatch correctly
    let cfg = EncoderConfig::from_model_name("answerdotai/ModernBERT-base");
    assert!(cfg.use_rope);
    assert_eq!(cfg.num_hidden_layers, 22);

    let cfg = EncoderConfig::from_model_name("answerdotai/ModernBERT-large");
    assert_eq!(cfg.num_hidden_layers, 28);

    let cfg = EncoderConfig::from_model_name("microsoft/deberta-v3-base");
    assert!(!cfg.use_rope);
    assert_eq!(cfg.vocab_size, 128100);

    // A BERT-like model without known keywords
    let cfg = EncoderConfig::from_model_name("google-bert/bert-base-uncased");
    assert_eq!(cfg.vocab_size, 30522);
    assert!(!cfg.use_rope);
}

// =========================================================================
// from_model_name: embedded substrings and edge cases
// =========================================================================

#[test]
fn test_from_model_name_substring_matching() {
    // "modernbert" embedded in a larger string still matches
    let cfg = EncoderConfig::from_model_name("my-finetuned-modernbert-v2");
    assert!(cfg.use_rope);
    assert_eq!(cfg.num_hidden_layers, 22); // base (no "large")

    // "deberta" embedded in a custom path
    let cfg = EncoderConfig::from_model_name("custom/deberta-finetuned");
    assert_eq!(cfg.vocab_size, 128100);
    assert!(!cfg.use_rope);

    // "large" without a known architecture still falls back to BERT
    let cfg = EncoderConfig::from_model_name("some-large-model");
    assert_eq!(cfg.vocab_size, 30522); // BERT fallback

    // Empty string falls back to BERT
    let cfg = EncoderConfig::from_model_name("");
    assert_eq!(cfg.vocab_size, 30522);
}

// =========================================================================
// Architecture: config round-trip consistency
// =========================================================================

#[test]
fn test_architecture_config_matches_max_length() {
    // EncoderArchitecture::max_length() should agree with its default config
    for arch in [
        EncoderArchitecture::Bert,
        EncoderArchitecture::DeBertaV3,
        EncoderArchitecture::ModernBert,
    ] {
        assert_eq!(
            arch.max_length(),
            arch.default_config().max_position_embeddings,
            "max_length() disagrees with default_config() for {}",
            arch
        );
    }
}

// =========================================================================
// Architecture: uses_rope agrees with config
// =========================================================================

#[test]
fn test_architecture_uses_rope_matches_config() {
    for arch in [
        EncoderArchitecture::Bert,
        EncoderArchitecture::DeBertaV3,
        EncoderArchitecture::ModernBert,
    ] {
        assert_eq!(
            arch.uses_rope(),
            arch.default_config().use_rope,
            "uses_rope() disagrees with default_config().use_rope for {}",
            arch
        );
    }
}

// =========================================================================
// Config: Clone produces identical values
// =========================================================================

#[test]
fn test_config_clone_identity() {
    let original = EncoderConfig::modernbert_large();
    let cloned = original.clone();
    assert_eq!(original.vocab_size, cloned.vocab_size);
    assert_eq!(original.hidden_size, cloned.hidden_size);
    assert_eq!(original.num_attention_heads, cloned.num_attention_heads);
    assert_eq!(original.num_hidden_layers, cloned.num_hidden_layers);
    assert_eq!(original.intermediate_size, cloned.intermediate_size);
    assert_eq!(
        original.max_position_embeddings,
        cloned.max_position_embeddings
    );
    assert_eq!(original.hidden_dropout_prob, cloned.hidden_dropout_prob);
    assert!((original.layer_norm_eps - cloned.layer_norm_eps).abs() < f64::EPSILON);
    assert_eq!(original.use_rope, cloned.use_rope);
    assert_eq!(original.use_geglu, cloned.use_geglu);
    assert!((original.rope_theta - cloned.rope_theta).abs() < f64::EPSILON);
    assert_eq!(original.use_pre_norm, cloned.use_pre_norm);
}
