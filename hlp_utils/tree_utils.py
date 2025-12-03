"""
HLP Utils Tree - Tree Transformation Operations

This module provides utilities for working with dependency trees,
including building, traversing, and transforming tree structures.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator, Set
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Node in a dependency tree"""
    id: str
    
    form: str
    
    lemma: Optional[str] = None
    
    pos: Optional[str] = None
    
    head: Optional[str] = None
    
    deprel: Optional[str] = None
    
    children: List[TreeNode] = field(default_factory=list)
    
    features: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: TreeNode):
        """Add a child node"""
        self.children.append(child)
    
    def get_descendants(self) -> List[TreeNode]:
        """Get all descendants"""
        descendants = []
        
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        
        return descendants
    
    def get_subtree_tokens(self) -> List[TreeNode]:
        """Get all tokens in subtree (including self)"""
        tokens = [self]
        tokens.extend(self.get_descendants())
        
        tokens.sort(key=lambda n: int(n.id) if n.id.isdigit() else 0)
        
        return tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "pos": self.pos,
            "head": self.head,
            "deprel": self.deprel,
            "children": [c.to_dict() for c in self.children],
            "features": self.features
        }


@dataclass
class DependencyTree:
    """Dependency tree structure"""
    root: Optional[TreeNode] = None
    
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    
    sentence_id: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_head(self, node_id: str) -> Optional[TreeNode]:
        """Get head of a node"""
        node = self.nodes.get(node_id)
        if not node or not node.head or node.head == "0":
            return None
        return self.nodes.get(node.head)
    
    def get_dependents(self, node_id: str) -> List[TreeNode]:
        """Get dependents of a node"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return node.children
    
    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """Get path from node to root"""
        path = []
        current = self.nodes.get(node_id)
        
        visited = set()
        
        while current and current.id not in visited:
            path.append(current)
            visited.add(current.id)
            
            if not current.head or current.head == "0":
                break
            
            current = self.nodes.get(current.head)
        
        return path
    
    def linearize(self) -> List[TreeNode]:
        """Linearize tree to token list"""
        tokens = list(self.nodes.values())
        tokens.sort(key=lambda n: int(n.id) if n.id.isdigit() else 0)
        return tokens
    
    def is_projective(self) -> bool:
        """Check if tree is projective"""
        edges = []
        
        for node in self.nodes.values():
            if node.head and node.head != "0":
                try:
                    dep_idx = int(node.id)
                    head_idx = int(node.head)
                    edges.append((min(dep_idx, head_idx), max(dep_idx, head_idx)))
                except ValueError:
                    continue
        
        for i, (s1, e1) in enumerate(edges):
            for s2, e2 in edges[i+1:]:
                if s1 < s2 < e1 < e2 or s2 < s1 < e2 < e1:
                    return False
        
        return True
    
    def get_depth(self) -> int:
        """Get maximum depth of tree"""
        if not self.root:
            return 0
        
        def node_depth(node: TreeNode) -> int:
            if not node.children:
                return 1
            return 1 + max(node_depth(c) for c in node.children)
        
        return node_depth(self.root)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "root": self.root.to_dict() if self.root else None,
            "node_count": len(self.nodes),
            "sentence_id": self.sentence_id,
            "is_projective": self.is_projective(),
            "depth": self.get_depth(),
            "metadata": self.metadata
        }


def build_dependency_tree(
    tokens: List[Dict[str, Any]],
    sentence_id: Optional[str] = None
) -> DependencyTree:
    """Build dependency tree from token list"""
    tree = DependencyTree(sentence_id=sentence_id)
    
    for token in tokens:
        node = TreeNode(
            id=str(token.get("id", "")),
            form=token.get("form", ""),
            lemma=token.get("lemma"),
            pos=token.get("pos") or token.get("upos"),
            head=str(token.get("head", "")) if token.get("head") is not None else None,
            deprel=token.get("deprel"),
            features=token.get("features", {})
        )
        tree.nodes[node.id] = node
    
    for node in tree.nodes.values():
        if node.head and node.head != "0":
            parent = tree.nodes.get(node.head)
            if parent:
                parent.add_child(node)
        elif node.head == "0" or not node.head:
            if tree.root is None:
                tree.root = node
    
    return tree


def linearize_tree(tree: DependencyTree) -> List[Dict[str, Any]]:
    """Linearize tree to token list"""
    tokens = []
    
    for node in tree.linearize():
        tokens.append({
            "id": node.id,
            "form": node.form,
            "lemma": node.lemma,
            "pos": node.pos,
            "head": node.head,
            "deprel": node.deprel,
            "features": node.features
        })
    
    return tokens


def find_head(
    tree: DependencyTree,
    node_id: str
) -> Optional[TreeNode]:
    """Find head of a node"""
    return tree.get_head(node_id)


def get_dependents(
    tree: DependencyTree,
    node_id: str
) -> List[TreeNode]:
    """Get dependents of a node"""
    return tree.get_dependents(node_id)


def get_subtree(
    tree: DependencyTree,
    node_id: str
) -> List[TreeNode]:
    """Get subtree rooted at a node"""
    node = tree.get_node(node_id)
    if not node:
        return []
    return node.get_subtree_tokens()


def is_projective(tree: DependencyTree) -> bool:
    """Check if tree is projective"""
    return tree.is_projective()


def find_common_ancestor(
    tree: DependencyTree,
    node_id1: str,
    node_id2: str
) -> Optional[TreeNode]:
    """Find lowest common ancestor of two nodes"""
    path1 = set(n.id for n in tree.get_path_to_root(node_id1))
    
    for node in tree.get_path_to_root(node_id2):
        if node.id in path1:
            return node
    
    return None


def get_siblings(
    tree: DependencyTree,
    node_id: str
) -> List[TreeNode]:
    """Get siblings of a node"""
    node = tree.get_node(node_id)
    if not node or not node.head:
        return []
    
    parent = tree.get_node(node.head)
    if not parent:
        return []
    
    return [c for c in parent.children if c.id != node_id]


def get_left_dependents(
    tree: DependencyTree,
    node_id: str
) -> List[TreeNode]:
    """Get dependents to the left of a node"""
    node = tree.get_node(node_id)
    if not node:
        return []
    
    try:
        node_idx = int(node_id)
    except ValueError:
        return []
    
    return [
        c for c in node.children
        if c.id.isdigit() and int(c.id) < node_idx
    ]


def get_right_dependents(
    tree: DependencyTree,
    node_id: str
) -> List[TreeNode]:
    """Get dependents to the right of a node"""
    node = tree.get_node(node_id)
    if not node:
        return []
    
    try:
        node_idx = int(node_id)
    except ValueError:
        return []
    
    return [
        c for c in node.children
        if c.id.isdigit() and int(c.id) > node_idx
    ]


def tree_to_conllu(tree: DependencyTree) -> str:
    """Convert tree to CoNLL-U format"""
    lines = []
    
    if tree.sentence_id:
        lines.append(f"# sent_id = {tree.sentence_id}")
    
    for node in tree.linearize():
        fields = [
            node.id,
            node.form,
            node.lemma or "_",
            node.pos or "_",
            "_",
            "_",
            node.head or "_",
            node.deprel or "_",
            "_",
            "_"
        ]
        lines.append("\t".join(fields))
    
    return "\n".join(lines)
