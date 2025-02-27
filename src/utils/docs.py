"""
Documentation generation utilities.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class APIDocGenerator:
    """Generates API documentation from code."""
    
    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def document_class(self, cls: Type) -> Dict[str, Any]:
        """Generate documentation for a class."""
        doc = {
            'name': cls.__name__,
            'docstring': inspect.getdoc(cls),
            'methods': [],
            'attributes': []
        }
        
        # Document methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_') or name == '__init__':
                method_doc = {
                    'name': name,
                    'docstring': inspect.getdoc(method),
                    'signature': str(inspect.signature(method)),
                    'async': inspect.iscoroutinefunction(method)
                }
                doc['methods'].append(method_doc)
        
        # Document attributes
        for name, value in inspect.getmembers(cls):
            if not name.startswith('_') and not callable(value):
                attr_doc = {
                    'name': name,
                    'type': str(type(value).__name__)
                }
                doc['attributes'].append(attr_doc)
        
        return doc
    
    def generate_markdown(self, doc: Dict[str, Any]) -> str:
        """Convert documentation dict to markdown."""
        lines = [
            f"# {doc['name']}",
            "",
            doc['docstring'] or "No description available.",
            "",
            "## Methods",
            ""
        ]
        
        for method in doc['methods']:
            lines.extend([
                f"### {method['name']}{method['signature']}",
                "",
                "**Async:** " + ("Yes" if method['async'] else "No"),
                "",
                method['docstring'] or "No description available.",
                ""
            ])
        
        if doc['attributes']:
            lines.extend([
                "## Attributes",
                ""
            ])
            for attr in doc['attributes']:
                lines.extend([
                    f"### {attr['name']}",
                    f"**Type:** {attr['type']}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def save_documentation(self, doc: Dict[str, Any], format: str = "markdown"):
        """Save documentation to file."""
        filename = f"{doc['name']}.{'md' if format == 'markdown' else 'json'}"
        file_path = self.output_dir / filename
        
        try:
            if format == "markdown":
                content = self.generate_markdown(doc)
                file_path.write_text(content)
            else:
                with open(file_path, 'w') as f:
                    json.dump(doc, f, indent=2)
                    
            logger.info(f"Documentation saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save documentation: {str(e)}")
            raise 