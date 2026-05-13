"""
Tests for Watermark Module
"""

import pytest
import torch


class TestWatermarkConfig:
    """Test WatermarkConfig initialization."""

    def test_default_config(self):
        from genGuard.model_layer.watermarker import WatermarkConfig
        config = WatermarkConfig()
        assert config.watermark_length == 50
        assert config.green_fraction == 0.25

    def test_custom_config(self):
        from genGuard.model_layer.watermarker import WatermarkConfig
        config = WatermarkConfig(green_fraction=0.3, watermark_length=100)
        assert config.green_fraction == 0.3
        assert config.watermark_length == 100


class TestGumbelWatermarker:
    """Test GumbelWatermarker functionality."""

    def test_initialization(self, watermarker):
        assert watermarker is not None
        assert watermarker.key is None
        assert watermarker.green_list is None

    def test_set_key(self, watermarker):
        watermarker.set_key(42)
        assert watermarker.key == 42
        assert watermarker.green_list is not None
        assert len(watermarker.green_list) > 0

    def test_green_list_size(self, watermarker):
        from genGuard.model_layer.watermarker import WatermarkConfig
        config = WatermarkConfig(green_fraction=0.25, vocab_size=50257)
        wm = GumbelWatermarker(config=config)
        wm.set_key(42)
        
        expected_size = int(50257 * 0.25)
        assert len(wm.green_list) == expected_size

    def test_generate_watermark_mask(self, watermarker):
        watermarker.set_key(42)
        
        mask = watermarker.generate_watermark_mask(10)
        
        assert mask.shape == (10,)
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_embed_watermark(self, watermarker):
        watermarker.set_key(42)
        
        logits = torch.randn(1, 5, 100)
        watermarked = watermarker.embed_watermark(logits)
        
        assert watermarked.shape == logits.shape
        assert not torch.equal(watermarked, logits)

    def test_detect_watermark_basic(self, watermarker):
        watermarker.set_key(42)
        
        score, detected = watermarker.detect_watermark("the world is beautiful")
        
        assert isinstance(score, float)
        assert isinstance(detected, bool)

    def test_watermark_words_generated(self, watermarker):
        watermarker.set_key(42)
        
        assert hasattr(watermarker, 'watermark_words')
        assert len(watermarker.watermark_words) > 0

    def test_detect_with_watermark_words(self, watermarker):
        watermarker.set_key(42)
        
        text_with_wm = " ".join(watermarker.watermark_words[:5])
        
        score, detected = watermarker.detect_watermark(text_with_wm)
        
        assert score > 0

    def test_detect_without_watermark(self, watermarker):
        watermarker.set_key(42)
        
        normal_text = "the world is beautiful and the sun is bright"
        
        score, detected = watermarker.detect_watermark(normal_text)
        
        assert isinstance(score, float)

    def test_key_reproducibility(self, watermark_config):
        wm1 = GumbelWatermarker(config=watermark_config)
        wm2 = GumbelWatermarker(config=watermark_config)
        
        wm1.set_key(42)
        wm2.set_key(42)
        
        assert wm1.green_list == wm2.green_list
        assert wm1.watermark_words == wm2.watermark_words

    def test_different_keys_different_green_list(self, watermark_config):
        wm1 = GumbelWatermarker(config=watermark_config)
        wm2 = GumbelWatermarker(config=watermark_config)
        
        wm1.set_key(42)
        wm2.set_key(123)
        
        assert wm1.green_list != wm2.green_list


class TestWatermarkIntegration:
    """Integration tests for watermark in pipeline."""

    def test_pipeline_watermark_embed(self, pipeline, sample_texts):
        result = pipeline.process_output(
            text=sample_texts["no_pii"],
            watermark=True
        )
        
        assert result is not None
        assert "output" in result

    def test_pipeline_watermark_detect(self, pipeline, watermarker, sample_texts):
        result = pipeline.process_output(
            text=sample_texts["no_pii"],
            watermark=True
        )
        
        watermarked_text = result["output"]
        score, detected = watermarker.detect_watermark(watermarked_text)
        
        assert isinstance(score, float)
        assert isinstance(detected, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])