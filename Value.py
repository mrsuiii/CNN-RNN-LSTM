from typing import Callable, Union, Tuple, Set
from graphviz import Digraph
import numpy as np

class Value:
    def __init__(self, data: np.ndarray | list):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self.label = None
        self._op = None
    def __getitem__(self, idx):
        # Pastikan hasil slicing selalu 2D
        sliced = self.data[idx]
        if sliced.ndim == 1:
            sliced = np.atleast_2d(sliced)
        return Value(sliced)
    def __repr__(self) :
        return str(self.data)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad

        # Topological sort to determine correct backward order
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + out.grad
        out._backward = _backward
        out._op = '+'
        return out
    
    def __radd__(self, other) :
        return self + other

    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other) :
        return (-self) + other
    
    def unbroadcast(self, grad, shape):
        """
        Reduces grad sehingga memiliki bentuk yang sama dengan `shape`.
        Fungsi ini melakukan penjumlahan (sum) pada axis yang tidak sesuai.
        """
        # Jika grad memiliki dimensi lebih dari shape yang diinginkan, jumlahkan sumbu ekstra
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # Periksa tiap axis, jika tidak sesuai, jumlahkan pada axis tersebut
        for i, dim in enumerate(shape):
            if grad.shape[i] != dim:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}
        def _backward():
            # Hitung gradien dasar untuk masing-masing operand
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            # Unbroadcast gradien sesuai dengan shape asli dari self.data dan other.data
            grad_self = self.unbroadcast(grad_self, self.data.shape)
            grad_other = self.unbroadcast(grad_other, other.data.shape)
            
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_self
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + grad_other
        out._backward = _backward
        out._op = '*'
        return out
    
    def __rmul__(self, other) :
        return self.__mul__(other)
    
    def reciprocal(self):
        out = Value(1.0 / self.data)
        out._prev = {self}
        
        def _backward():
            grad_input = -out.data * out.data * out.grad
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_input

        out._backward = _backward
        out._op = 'reciprocal'
        return out
    
    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return Value(other) * self.reciprocal()
    
    def __gt__(self, other):
        
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(np.where(self.data > other.data, 1.0, 0.0))
        out._prev = {self, other}
        
        def _backward():
            pass
        
        out._backward = _backward
        out._op = '>'
        return out
    
    def __len__(self):
        return len(self.data)
    
    def matmul(self, other):
        out = Value(self.data.dot(other.data))
        out._prev = {self, other}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.dot(other.data.T)
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data.T.dot(out.grad)
        out._backward = _backward
        out._op = 'matmul'
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data))
        out._prev = {self}
        def _backward():
            grad = (self.data > 0).astype(np.float32)
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad * out.grad
        out._backward = _backward
        out._op = "relu"
        return out

    def log(self):
        eps = 1e-7
        out = Value(np.log(np.maximum(self.data, eps)))
        out._prev = {self}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (1 / np.maximum(self.data, eps)) * out.grad
        out._backward = _backward
        out._op = 'log'
        return out

    def sqrt(self):
        out = Value(np.sqrt(self.data))
        out._prev = {self}
        
        def _backward():
            grad_input = (0.5 / out.data) * out.grad
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_input

        out._backward = _backward
        out._op = 'sqrt'
        return out

    def __neg__(self):
        return self * (-1)

    def transpose(self):
        out = Value(self.data.T)
        out._prev = {self}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.T
        out._backward = _backward
        out._op = 'transpose'
        return out
    
    def abs(self):
        out = Value(np.abs(self.data))
        out._prev = {self}
        
        def _backward():
            grad_input = np.sign(self.data) * out.grad
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_input

        out._backward = _backward
        out._op = 'abs'
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Value(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            grad_input = np.ones_like(self.data) * out.grad
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad_input

        out._backward = _backward
        out._op = 'sum'
        return out

    @property
    def T(self):
        return self.transpose()

def mean(t: Value) -> Value:
    data = np.mean(t.data)
    out = Value(data)
    out._prev = {t}
    def _backward():
        grad = np.ones_like(t.data) * (1 / t.data.size) * out.grad
        t.grad = (t.grad if t.grad is not None else np.zeros_like(t.data)) + grad
    out._backward = _backward
    out._op = 'mean'
    return out

# Fungsi sum sepanjang axis tertentu (misalnya axis=1 untuk CCE)
def sum_axis(t: Value, axis: int) -> Value:
    data = np.sum(t.data, axis=axis, keepdims=True)
    out = Value(data)
    out._prev = {t}
    def _backward():
        grad = out.grad * np.ones_like(t.data)
        t.grad = (t.grad if t.grad is not None else np.zeros_like(t.data)) + grad
    out._backward = _backward
    out._op = 'sum_axis'
    return out

def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    nodes, edges = set(), set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        name = str(id(n))
        label = n.label if hasattr(n, "label") and n.label else "const " + n.__repr__()  
        dot.node(name, label=f"{label} | data {str(n.data)} | grad {str(n.grad)}", shape='record')
        if hasattr(n, '_op') and n._op:  
            op_name = name + n._op  
            dot.node(op_name, label=n._op, shape='circle')
            dot.edge(op_name, name)  

    for n1, n2 in edges:
        name1 = str(id(n1))
        name2 = str(id(n2))
        if hasattr(n2, '_op') and n2._op:
            op_name = name2 + n2._op  
            dot.edge(name1, op_name)  
        else:
            dot.edge(name1, name2)  

    return dot

if __name__ == "__main__":
    np.random.seed(42)

    A = Value(np.random.randn(3, 3))
    B = Value(np.random.randn(3, 3))
    A.label = 'A'
    B.label = 'B'

    C = A.matmul(B)
    C.label = 'C = A @ B'

    D = C + Value(np.ones_like(C.data) * 2)
    D.label = 'D = C + 2'

    E = D.relu()
    E.label = 'E = ReLU(D)'

    F = E.log()
    F.label = 'F = log(E)'

    G = mean(F)
    G.label = 'G = mean(F)'

    H = sum_axis(F, axis=1)
    H.label = 'H = sum_axis(F, axis=1)'

    G.backward()

    dot = draw_dot(G)
    dot.render("computation_graph", format="png", cleanup=True)

