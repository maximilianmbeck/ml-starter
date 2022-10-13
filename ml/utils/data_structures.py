import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Generic, Iterable, List, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BiDict(Dict[K, V], Generic[K, V]):
    """Defines a dictionary which can be used to do bi-directional lookups.

    This is essentially the same as a regular dictionary but with O(1) value
    lookups (i.e., can immediately look up a value instead of searching the
    whole dictionary).
    """

    def __init__(self, items: Dict[K, V] | None = None) -> None:
        if items is None:
            super().__init__()
        else:
            super().__init__(**items)

        self.vk = {v: k for k, v in self.items()}

    def __setitem__(self, k: K, v: V) -> None:
        super().__setitem__(k, v)

        self.vk[v] = k

    def pop(self, k: K) -> V:  # type: ignore
        v = super().pop(k)
        assert self.vk.pop(v) == k
        return v

    def get_value(self, v: V) -> K:
        return self.vk[v]

    def set_value(self, v: V, k: K) -> None:
        super().__setitem__(k, v)

        self.vk[v] = k

    def pop_value(self, v: V) -> K:
        k = self.vk.pop(v)
        assert super().pop(k) == v
        return k


class PriorityQueue(Generic[K, V]):
    """Defines a wrapper class around underlying heap implementation.

    I don't really like trying to remember how to use the heapq API every time
    I want to use a heap, so it's easier to have this wrapper class instead.
    """

    def __init__(self, items: Iterable[Tuple[K, V]] | None = None) -> None:
        self.heap: List[Tuple[K, int, V]]
        if items is None:
            self.heap = []
        else:
            self.heap = [(k, i, v) for i, (k, v) in enumerate(items)]
            heapq.heapify(self.heap)

    def clear(self) -> None:
        """Clears the priority queue."""

        self.heap.clear()

    def empty(self) -> bool:
        """Returns if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise
        """

        return len(self.heap) == 0

    def push(self, priority: K, value: V) -> None:
        """Adds the key to the priority queue.

        Args:
            priority: The priority of the item to add (lower integer values
                have higher priority)
            value: The associated item
        """

        heapq.heappush(self.heap, (priority, len(self), value))

    def pop(self) -> Tuple[K, V]:
        """Pops the highest-priority / lowest-value item from the queue.

        Returns:
            The priority and value of the highest-priority element
        """

        k, _, v = heapq.heappop(self.heap)
        return k, v

    def peak(self) -> Tuple[K, V]:
        """Look at the highest-priority element without popping.

        Returns:
            The priority and value of the highest-priority element
        """

        k, _, v = self.heap[0]
        return k, v

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return not self.empty()


@dataclass
class Node(Generic[K]):
    val: K
    next: "Node[K]"
    prev: "Node[K]"


class LinkedList(Generic[K]):
    """Defines a simple doubly-connected linked list.

    Operations:

    - add: Add an element in O(1)
    - extend: Add an element after a given element in O(1)
    - pop: Remove an element in O(1)
    - swap: Swaps an old value with a new value in O(1), equivalent to doing an
        `extend` with the new element then a `pop` the old element
    - empty: If the list is empty in O(1)
    - to_list: Converts the linked list to a regular list in O(N)
    - positions: Indices of each item in the linked list in O(N)

    This data structure was necessary for a particular graph visualization
    algorithm, probably isn't the best choice for other algorithms.
    """

    def __init__(self) -> None:
        self.lookup: Dict[K, Node[K]] = {}
        self.head: Node[K] | None = None

    def add(self, k: K) -> None:
        node = Node(val=k, next=None, prev=None)  # type: ignore
        self.lookup[k] = node
        if self.head is None:
            self.head = node
            self.head.next = self.head
            self.head.prev = self.head
        else:
            node.next = self.head
            node.prev = self.head.prev
            node.prev.next = node
            node.next.prev = node

    def extend(self, k: K, new_k: K) -> None:
        node = self.__get_node(k)
        new_node = Node(val=new_k, next=node.next, prev=node)
        node.next.prev = new_node
        node.next = new_node
        self.lookup[new_k] = new_node

    def pop(self, k: K) -> None:
        node = self.lookup.pop(k)
        if node == self.head:
            self.head = node.next
        if len(self.lookup) == 0:
            self.head = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

    def swap(self, a: K, b: K) -> None:
        node = self.lookup.pop(a)
        node.val = b
        self.lookup[b] = node

    def empty(self) -> bool:
        return self.head is None

    def to_list(self) -> List[K]:
        items: List[K] = []
        if self.head is None:
            assert len(self) == 0
            return items
        node = self.head
        items.append(node.val)
        node = node.next
        while node != self.head:
            items.append(node.val)
            node = node.next
        assert len(self) == len(items)
        return items

    def positions(self) -> Dict[K, int]:
        return {k: i for i, k in enumerate(self.to_list())}

    def __get_node(self, k: K) -> Node[K]:
        node = self.lookup[k]
        assert node.val == k
        return node

    def __setitem__(self, k: K, v: K) -> None:
        self.swap(k, v)

    def __len__(self) -> int:
        return len(self.lookup)

    def __bool__(self) -> bool:
        return not self.empty()


def nested_defaultdict() -> DefaultDict[Any, Any]:
    constructor = lambda: defaultdict(constructor)  # type: ignore
    return constructor()
