"""
Prompt Manager

Handles system prompts, conversation history, and template management.
"""
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

from loguru import logger

@dataclass
class Message:
    """A message in the conversation."""
    role: str  # 'system', 'user', or 'assistant'
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=data.get('timestamp', datetime.now().timestamp()),
            metadata=data.get('metadata', {})
        )

class PromptManager:
    """Manages prompts, conversation history, and template rendering."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        max_tokens: int = 2048,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """Initialize the prompt manager.
        
        Args:
            system_prompt: Initial system prompt
            max_history: Maximum number of messages to keep in history
            max_tokens: Maximum number of tokens for the prompt
            token_counter: Function to count tokens in text
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_history = max(1, max_history)
        self.max_tokens = max(512, max_tokens)  # Minimum reasonable context
        self.token_counter = token_counter or self._dummy_token_counter
        self.messages: List[Message] = []
        self.templates: Dict[str, str] = {}
        self.variables: Dict[str, Any] = {}
        
        # Add system prompt if provided
        if self.system_prompt:
            self.add_message('system', self.system_prompt)
    
    def _default_system_prompt(self) -> str:
        """Default system prompt if none provided."""
        return (
            "You are a helpful, respectful, and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, "
            "sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature."
        )
    
    @staticmethod
    def _dummy_token_counter(text: str) -> int:
        """Dummy token counter that estimates based on whitespace."""
        return len(text.split())  # Rough estimate
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> 'PromptManager':
        """Add a message to the conversation history.
        
        Args:
            role: 'system', 'user', or 'assistant'
            content: Message content
            metadata: Additional metadata
            timestamp: Optional timestamp (default: current time)
            
        Returns:
            self for method chaining
        """
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'")
        
        message = Message(
            role=role,
            content=content,
            timestamp=timestamp or datetime.now().timestamp(),
            metadata=metadata or {}
        )
        
        # If adding a system message, replace the existing one
        if role == 'system':
            self.messages = [m for m in self.messages if m.role != 'system']
        
        self.messages.append(message)
        
        # Trim history if needed
        self._trim_history()
        
        return self
    
    def add_user_message(self, content: str, **kwargs) -> 'PromptManager':
        """Add a user message to the conversation."""
        return self.add_message('user', content, **kwargs)
    
    def add_assistant_message(self, content: str, **kwargs) -> 'PromptManager':
        """Add an assistant message to the conversation."""
        return self.add_message('assistant', content, **kwargs)
    
    def add_system_message(self, content: str, **kwargs) -> 'PromptManager':
        """Add or update the system message."""
        return self.add_message('system', content, **kwargs)
    
    def _trim_history(self):
        """Trim the conversation history to fit within token limits."""
        if not self.messages:
            return
        
        # Always keep system messages
        system_messages = [m for m in self.messages if m.role == 'system']
        other_messages = [m for m in self.messages if m.role != 'system']
        
        # If we're over the history limit, remove oldest non-system messages first
        while len(other_messages) > self.max_history:
            other_messages.pop(0)
        
        # Check token count and trim if needed
        current_tokens = sum(
            self.token_counter(m.content) 
            for m in system_messages + other_messages
        )
        
        while current_tokens > self.max_tokens and len(other_messages) > 1:
            # Remove the oldest non-system message (after the first one)
            if len(other_messages) > 1:  # Always keep at least one user message
                removed = other_messages.pop(0)
                current_tokens -= self.token_counter(removed.content)
            else:
                break
        
        # Rebuild messages with system messages first
        self.messages = system_messages + other_messages
    
    def render_template(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """Render a template with the given variables.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Variables to use in the template
            
        Returns:
            Rendered template string
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        
        # Merge instance variables with provided kwargs
        variables = {**self.variables, **kwargs}
        
        # Simple template rendering with {variable} syntax
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            # Try to render with available variables
            return template.format(
                **{k: v for k, v in variables.items() if f"{{{k}}}" in template}
            )
    
    def load_templates(self, templates: Dict[str, str]) -> 'PromptManager':
        """Load templates from a dictionary.
        
        Args:
            templates: Dictionary of template names to template strings
            
        Returns:
            self for method chaining
        """
        self.templates.update(templates)
        return self
    
    def load_templates_from_file(self, file_path: Union[str, Path]) -> 'PromptManager':
        """Load templates from a JSON file.
        
        Args:
            file_path: Path to JSON file containing templates
            
        Returns:
            self for method chaining
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            self.templates.update(templates)
            logger.info(f"Loaded {len(templates)} templates from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load templates from {file_path}: {e}")
            raise
        
        return self
    
    def set_variables(self, **kwargs) -> 'PromptManager':
        """Set template variables.
        
        Args:
            **kwargs: Variables to set
            
        Returns:
            self for method chaining
        """
        self.variables.update(kwargs)
        return self
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the current conversation history as a list of message dicts."""
        return [m.to_dict() for m in self.messages]
    
    def get_conversation_text(self, include_metadata: bool = False) -> str:
        """Get the conversation as formatted text.
        
        Args:
            include_metadata: Whether to include metadata in the output
            
        Returns:
            Formatted conversation text
        """
        lines = []
        
        for msg in self.messages:
            role = msg.role.upper()
            content = msg.content
            
            if include_metadata and msg.metadata:
                meta = ' '.join(f"{k}={v}" for k, v in msg.metadata.items())
                lines.append(f"[{role} | {meta}]\n{content}")
            else:
                lines.append(f"[{role}]\n{content}")
            
            lines.append("\n" + ("-" * 40) + "\n")
        
        return "\n".join(lines).strip()
    
    def clear_history(self, keep_system: bool = True) -> 'PromptManager':
        """Clear the conversation history.
        
        Args:
            keep_system: Whether to keep system messages
            
        Returns:
            self for method chaining
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == 'system']
        else:
            self.messages = []
        return self
    
    def save_conversation(self, file_path: Union[str, Path]) -> None:
        """Save the conversation to a file.
        
        Args:
            file_path: Path to save the conversation to
        """
        try:
            data = {
                'system_prompt': self.system_prompt,
                'messages': [m.to_dict() for m in self.messages],
                'templates': self.templates,
                'variables': self.variables,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'message_count': len(self.messages)
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved conversation to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    @classmethod
    def load_conversation(
        cls,
        file_path: Union[str, Path],
        **kwargs
    ) -> 'PromptManager':
        """Load a conversation from a file.
        
        Args:
            file_path: Path to the conversation file
            **kwargs: Additional arguments for the PromptManager constructor
            
        Returns:
            Loaded PromptManager instance
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create a new instance
            manager = cls(
                system_prompt=data.get('system_prompt'),
                **kwargs
            )
            
            # Load messages
            manager.messages = [
                Message.from_dict(msg) 
                for msg in data.get('messages', [])
            ]
            
            # Load templates and variables
            manager.templates = data.get('templates', {})
            manager.variables = data.get('variables', {})
            
            logger.info(f"Loaded conversation from {file_path} with {len(manager.messages)} messages")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of the conversation."""
        return self.get_conversation_text()
    
    def __len__(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)
    
    def __bool__(self) -> bool:
        """Whether there are any messages in the conversation."""
        return bool(self.messages)
