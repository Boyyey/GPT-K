"""Configuration settings for the AI assistant."""
from pathlib import Path
from typing import Dict, Any, Optional
import os
import json

class Config:
    """Configuration manager for the AI assistant."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Default configuration
        self._config = {
            "app": {
                "name": "GPT-K",
                "version": "0.1.0",
                "environment": os.getenv("APP_ENV", "development"),
                "debug": os.getenv("DEBUG", "false").lower() == "true",
            },
            "memory": {
                "db_url": os.getenv("MEMORY_DB_URL", "sqlite:///memories.db"),
                "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "memories.faiss"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
                "cleanup_days": int(os.getenv("CLEANUP_DAYS", "30")),
                "min_importance": float(os.getenv("MIN_IMPORTANCE", "0.3")),
            },
            "llm": {
                "model_name": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "2048")),
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "file": os.getenv("LOG_FILE", "app.log"),
                "rotation": os.getenv("LOG_ROTATION", "10 MB"),
                "retention": os.getenv("LOG_RETENTION", "30 days"),
            },
        }
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            self.load(config_path)
    
    def load(self, config_path: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            self._deep_update(self._config, config_data)
    
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'app.name')
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using bracket notation."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return self._config.copy()

# Global configuration instance
config = Config()
