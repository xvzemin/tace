"""Provides classes and functions for O(3) and SO(3) irreducible representations (irreps)."""
import math
import re
from typing import List, Literal, Tuple, Union, overload

import torch

p_mapping = {
    'e': +1, 'o': -1, +1: +1, -1: -1, 0: 0, None: 0
}

class Irrep:
    r"""
    An irreducible representation (irrep) of O(3) (when parity is ``'e'`` or ``'o'``) 
    or SO(3) (when the parity is None) group.

    An ``Irrep`` can be created with the following ways:
    
    >>> Irrep(2, 'e')   # O(3) irrep l=2 with even parity
    2e
    >>> Irrep("3o")     # O(3) irrep l=3 with odd parity  
    3o
    >>> Irrep(1)        # SO(3) vector irrep (p=None)
    1
    >>> Irrep((2, -1))  # Using numeric parity values
    2o

    Attributes:
        l (int): Rotation order/degree.
        p (int): Parity component (-1, +1, or 0 for None).
        dim (int): Dimensionality of the irrep (:math:`2l + 1`).
    """
    
    @overload
    def __init__(self, l: int, p: Union[Literal['e', 'o'], int, None] = None): ...
    
    @overload
    def __init__(self, lp: Union[str, Tuple[int, Union[Literal['e', 'o'], int, None]]]): ...
    
    def __init__(self, *args):
        # Check the number of arguments
        if len(args) == 0:
            raise TypeError("Irrep requires at least one argument")
        
        # If two arguments are passed, they must be (l, p)
        if len(args) == 2:
            l, p = args
            if not isinstance(l, int) or p is None:
                raise TypeError("Cannot pass both string and (l, p) initialization")
            self._init_from_lp(l, p)
        
        # If one argument is passed, it could be a string or l
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                self._init_from_str(arg)
            elif isinstance(arg, int):
                self._init_from_lp(arg, None)
            elif isinstance(arg, tuple):
                self._init_from_lp(*arg)
            else:
                raise TypeError("Invalid argument type")
        
        else:
            raise TypeError("Irrep accepts at most two arguments")

    def _init_from_lp(self, l: int, p: Union[Literal['e', 'o'], int, None]):
        """Initialize from (l, p)."""
        if p not in p_mapping:
            raise ValueError("p must be +1 ('e'), -1 ('o'), or 0 (None)")
        if not isinstance(l, int) or l < 0:
            raise ValueError("l must be a non-negative integer")
        self.p = p_mapping[p]
        self.l = l

    def _init_from_str(self, lp: str):
        """Initialize from string."""
        if not isinstance(lp, str):
            raise TypeError("Expected a string for initialization")
        
        # Parse the string
        if lp[-1] == 'e':
            l = int(lp[:-1])
            p = 'e'
        elif lp[-1] == 'o':
            l = int(lp[:-1])
            p = 'o'
        else:
            l = int(lp)
            p = None
        
        self._init_from_lp(l, p)

    def __iter__(self):
        """Supports unpacking, e.g., :math:`l, p = \text{irrep}`."""
        yield self.l
        yield self.p

    def __repr__(self):
        if self.p == 0:
            return str(self.l)
        else:
            return str(self.l)+'e' if self.p>0 else str(self.l)+'o' 

    def __eq__(self, other):
        if isinstance(other, str):
            return self.__repr__() == other
        elif isinstance(other, int):
            return self.l == other and self.p==0
        elif not isinstance(other, Irrep) and not isinstance(other, Tuple):
            return False
        else:
            l, p = other
            p = p_mapping[p]
            l = int(l)
            return self.p == p and self.l == l

    def __hash__(self):
        return hash((self.l, self.p))
    
    @property
    def dim(self):
        return 2 * self.l + 1
    
    def __mul__(self, other: 'Irrep') -> 'Irreps':
        r"""
        Compute the tensor product decomposition of two O(3) / SO(3) irreducible representations.

        Args:
            other (Irrep): The other irreducible representation to compute the tensor product with.

        Returns:
            Irreps: Direct sum of irreducible representations resulting from the tensor product.
            Contains all possible irreducible representations with degree :math:`l`
            in the range :math:`|l_1 - l_2| \leq l \leq l_1 + l_2` and parity :math:`p = p_1 p_2`. 
            Each irreducible representation has multiplicity 1.

        Raises:
            TypeError: If ``other`` is not an instance of ``Irrep``.

        Examples:
            >>> Irrep("1e") * Irrep("1o")
            Irreps(0o+1o+2o)
            >>> Irrep("2e") * Irrep("1e")
            Irreps(1e+2e+3e)
            >>> Irrep("0o") * Irrep("0o")
            Irreps(0e)
            >>> Irrep("3e") * Irrep("0")
            Irreps(3)
            >>> Irrep("1") * Irrep("3o")
            Irreps(2+3+4)
        """
        if not isinstance(other, Irrep):
            raise TypeError(f"Can only multiply Irrep with Irrep, got {type(other)}")

        return Irreps([
            Irrep(l, self.p * other.p) for l in range(abs(self.l-other.l), self.l+other.l+1)
        ])


def parse_irreps(str_irreps: str) -> List[Tuple[Irrep, int]]:
    """
    Parse an irreducible representation string and return a list of ``(Irrep, multiplicity)``.
    
    Args:
        str_irreps (str): The string representation of irreps.

    Returns:
        list[tuple[Irrep, int]]: A list of (Irrep, multiplicity) pairs.

    Example:
        >>> parse_irreps('6x0+1+2e+5x2o+3x3e')
        [(0, 6), (1, 1), (2e, 1), (2o, 5), (3e, 3)]
    """
    # Regular expression pattern
    pattern = re.compile(r'(\d+x)?([0-9]+[eo]?)')
    
    # Parsing result
    result = []
    str_irreps = str_irreps.replace(' ','')
    # Split the string by '+'
    for term in str_irreps.split('+'):
        # Remove whitespace
        term = term.strip()
        if not term:
            continue
        
        # Match the regular expression
        match = pattern.fullmatch(term)
        if not match:
            raise ValueError(f"Invalid term: {term}")
        
        # Extract multiplicity and irrep string
        multiplicity_str, irrep_str = match.groups()
        
        # Handle multiplicity
        multiplicity = int(multiplicity_str[:-1]) if multiplicity_str else 1

        if multiplicity < 1:  # Check validity
            raise ValueError(f"Multiplicity must be \u22651 in '{term}'")
        
        # Create Irrep object
        irrep = Irrep(irrep_str)
        
        # Add to the result list
        result.append((irrep, multiplicity))
    return result

def show_irreps(list_irreps: List[Tuple[Irrep, int]]) -> str:
    """Convert a list of (Irrep, multiplicity) to string representation."""
    return '+'.join([f'{"" if ir[1] == 1 else f"{ir[1]}x"}{ir[0]}' for ir in list_irreps])

def check_irreps(irreps: Union[str, Tuple, List, 'Irrep', 'Irreps']) -> 'Irreps':
    """Ensure the input is an Irreps object."""
    if isinstance(irreps, Irreps):
        return irreps
    else:
        return Irreps(irreps)
    
def has_path(irrep_out: Irrep, irrep_1: Irrep, irrep_2: Irrep = None) -> bool:
    """
    Check if a tensor product path exists.

    Args:
        irrep_out (Irrep): The output irrep.
        irrep_1 (Irrep): The first input irrep.
        irrep_2 (Irrep, optional): The second input irrep. Defaults to None.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    if irrep_2 is not None:
        return (
            (abs(irrep_1.l-irrep_2.l) <= irrep_out.l <= irrep_1.l+irrep_2.l)
            and (irrep_out.p == 0 or irrep_1.p * irrep_2.p == irrep_out.p) 
        )
    else:
        return (irrep_out.l == irrep_1.l) and (irrep_out.p == 0 or irrep_1.p == irrep_out.p)

class Irreps:
    """
    Collection of irreducible representations with multiplicities.

    An ``Irreps`` can be created with the following ways:
    
    >>> Irreps("6x0 + 1 + 2e + 5x2o")  # Explicit O(3) irreps
    Irreps(6x0+1+2e+5x2o)
    >>> Irreps((0, 3))                 # SO(3) irreps of l=0 to 3
    Irreps(0+1+2+3)
    >>> Irreps([0, "1e", "2o"])        # Mixed SO(3) and O(3) irreps
    Irreps(0+1e+2o)
    >>> Irreps("1e")                   # Single irrep
    Irreps(1e)

    Attributes:
        irrep_groups (list[tuple[Irrep, int]]): List of irrep-multiplicity pairs.
    """

    irrep_groups: List[Tuple[Irrep, int]]

    @staticmethod
    def spherical_harmonics(max_degree: int, parity_even_degree: Union[Literal['e', 'o'], None] = 'e') -> 'Irreps':
        """Generate irreps with alternating parities based on degree.
        
        Args:
            max_degree (int): Maximum degree (:math:`l`) of the irreps to generate.
            parity_even_degree (str, optional): Parity to assign to even degrees. 
                Can be 'e' or 'o'. Defaults to 'e'.
            
        Returns:
            Irreps: Irreps with alternating parities from :math:`l=0` to :math:`l=\text{max_degree}`.
            
        Examples:
            >>> Irreps.spherical_harmonics(2, 'e')
            Irreps(0e+1o+2e)
            >>> Irreps.spherical_harmonics(3, 'o')
            Irreps(0o+1e+2o+3e)
        """
        if parity_even_degree == 'e':
            return Irreps([f"{l}{'e' if l%2==0 else 'o'}" for l in range(max_degree+1)])
        elif parity_even_degree == 'o':
            return Irreps([f"{l}{'o' if l%2==0 else 'e'}" for l in range(max_degree+1)])
        else:
            raise ValueError("parity_even_degree must be 'e' or 'o'")
        
    @staticmethod
    def both_parities(max_degree: int, min_degree: int = 0) -> 'Irreps':
        """Generate irreps with both parities (even and odd) for each degree in a range.
        
        For each degree :math:`l` from ``min_degree`` to ``max_degree`` (inclusive), this method creates
        two irreps: one with even parity (e) and one with odd parity (o). This is useful
        when you need to consider both possible parities for each degree in a specified range.
        
        Args:
            max_degree (int): Maximum degree (:math:`l`) of the irreps to generate.
            min_degree (int, optional): Minimum degree (:math:`l`) to start from. Defaults to 0.
            
        Returns:
            Irreps: Irreps containing both even and odd parity versions for each degree
            from :math:`l=\text{min_degree}` to :math:`l=\text{max_degree}`.
            
        Examples:
            >>> Irreps.both_parities(1)
            Irreps(0e+0o+1e+1o)
            >>> Irreps.both_parities(2, min_degree=1)
            Irreps(1e+1o+2e+2o)
            >>> Irreps.both_parities(3, min_degree=2)
            Irreps(2e+2o+3e+3o)
        """
        return Irreps('+'.join([f"{l}e+{l}o" for l in range(min_degree, max_degree+1)]))
    
    def __init__(self, irreps: Union[str, Tuple, List, Irrep, 'Irreps']):
        if isinstance(irreps, str):
            self.irrep_groups = parse_irreps(irreps)
        elif isinstance(irreps, tuple):
            if len(irreps) == 2 and isinstance(irreps[0], int) and isinstance(irreps[1], int):
                start, end = irreps
                self.irrep_groups = [(Irrep(ir), 1) for ir in range(start, end + 1)]
            elif len(irreps) == 2 and isinstance(irreps[0], Irrep) and isinstance(irreps[1], int):
                self.irrep_groups = [irreps]
            else:
                raise ValueError("Invalid tuple format. Expected two integers for range.")
        elif isinstance(irreps, list):
            self.irrep_groups = [
                (ir, 1) if isinstance(ir, Irrep)
                else (ir[0], ir[1]) if isinstance(ir, tuple) and isinstance(ir[0], Irrep)
                else (Irrep(ir), 1) for ir in irreps
            ]
        elif isinstance(irreps, Irrep):
            self.irrep_groups = [(irreps, 1)]
        elif isinstance(irreps, Irreps):
            self.irrep_groups = parse_irreps(irreps.short_repr())
        else:
            raise TypeError(f"Invalid type for irrep_groups: {type(irreps)}")

    def __repr__(self):
        return f"Irreps({show_irreps(self.irrep_groups)})"

    def __eq__(self, other):
        """
        Check equality between two ``Irreps`` instances.

        Two ``Irreps`` are considered equal if and only if:
        - Their ``irrep_groups`` are identical as lists

        If ``other`` is not an ``Irreps`` instance, it will first be 
        converted to one using ``Irreps(other)``.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if equal, False otherwise.

        Examples:
            >>> Irreps("1x0e") == Irreps("1x0e")
            True
            >>> Irreps("1x0e") == Irreps("0e")
            True
            >>> Irreps("1x0e") == Irreps("1x0")
            False
            >>> Irreps("2x0e+1x0e") == Irreps("3x0e")
            False
            >>> Irreps("2x0e+1x0e") == Irreps("1x0e+2x0e")
            False
            >>> Irreps("3x1e+2o") == "3x1e + 2o"
            True
        """
        if other is None:
            return False
        other = check_irreps(other)
        return self.irrep_groups == other.irrep_groups
    
    def sorted(self, by: callable = None) -> 'Irreps':
        """
        Returns a new Irreps instance with sorted ``irrep_groups``.

        The default sorting key orders irreps by their degree (:math:`l`), parity (None < 'e' < 'o'), 
        and multiplicity. This ensures a consistent and intuitive ordering.
        
        Args:
            by (callable, optional): A function to use for sorting. Defaults to None.

        Returns:
            Irreps: A new sorted Irreps instance.

        Examples:
            >>> Irreps("0e+0o+0").sorted()
            Irreps(0+0e+0o)
            >>> Irreps("3x0e+0e").sorted()
            Irreps(0e+3x0e)
        """
        if by is None:
            by = lambda ir_mul: (
                ir_mul[0].l, 
                ir_mul[0].p != 0, 
                ir_mul[0].p != 1, 
                ir_mul[0].p != -1, 
                ir_mul[1]
            )
        sorted_pairs = sorted(self.irrep_groups, key=by)
        return Irreps(sorted_pairs)

    def merged(self) -> 'Irreps':
        """
        Returns a new Irreps instance with consecutive identical irreps merged.
        Preserves the original order and only merges adjacent duplicates.

        Returns:
            Irreps: A new Irreps instance with merged consecutive irreps.

        Examples:
            >>> Irreps("0e+0e+1o+2e+2e+2e").merged()
            Irreps(2x0e+1o+3x2e)
            >>> Irreps("0+0e+1o+0e+0e").merged()  # Non-consecutive 0e are not merged
            Irreps(0e+1o+2x0e)
        """
        if not self.irrep_groups:
            return Irreps([])
        
        merged_groups = []
        current_irrep, current_mul = self.irrep_groups[0]
        
        for irrep, mul in self.irrep_groups[1:]:
            if irrep == current_irrep:
                current_mul += mul
            else:
                merged_groups.append((current_irrep, current_mul))
                current_irrep, current_mul = irrep, mul
        
        merged_groups.append((current_irrep, current_mul))
        return Irreps(merged_groups)

    def simplified(self) -> 'Irreps':
        """
        Returns a new Irreps instance with merged multiplicities of identical irreps.

        Combines all occurrences of the same irrep into a single entry with summed multiplicities. 
        The result is sorted to maintain a consistent order.
        
        Returns:
            Irreps: A new simplified and sorted Irreps instance.

        Examples:
            >>> Irreps("0o+3x0o+0e+3x1+3x2e+4x1").simplified()
            Irreps(0e+4x0o+7x1+3x2e)
        """
        counts = {}
        for irrep, mul in self.irrep_groups:
            counts[irrep] = counts.get(irrep, 0) + mul
        merged = [(irrep, mul) for irrep, mul in counts.items()]
        return Irreps(merged).sorted()
    
    def __add__(self, other: Union[str, Tuple, List, Irrep, 'Irreps']) -> 'Irreps':
        """
        Concatenate two Irreps objects.

        Args:
            other (Irreps or convertible): The other Irreps to add. 
                Can be any type convertible to Irreps via ``Irreps(...)``.

        Returns:
            Irreps: New Irreps containing combined ``irrep_groups`` from both inputs.

        Examples:
            >>> Irreps("2e") + Irreps("1o")
            Irreps(2e+1o)
            >>> "3e" + Irreps("2e")
            Irreps(3e+2e)
        """
        other = check_irreps(other)
        return Irreps(self.irrep_groups + other.irrep_groups)

    def __radd__(self, other: Union[str, Tuple, List, Irrep, 'Irreps']) -> 'Irreps':
        """Right addition for Irreps objects."""
        other = check_irreps(other)
        return Irreps(other.irrep_groups + self.irrep_groups)

    def eliminate_parity(self, simplified: bool = False) -> 'Irreps':
        """
        Converts O(3) irreducible representations to SO(3) by eliminating parities.

        This method transforms the irreps list by setting the parity to 0 for 
        all representations, projecting them from the O(3) group to the SO(3) group.

        Args:
            simplified (bool, optional): If True, simplify the resulting Irreps. Defaults to False.

        Returns:
            Irreps: New set of irreducible representations in SO(3) format where all
            parities are 0.

        Example:
            >>> Irreps("4x0o+4x0e+4x1o+4x1e").eliminate_parity()
            Irreps(4x0+4x0+4x1+4x1)
            >>> Irreps("4x0o+4x0e+4x1o+4x1e").eliminate_parity(simplified=True)
            Irreps(8x0+8x1)
        """

        eliminated =  Irreps(
            [(Irrep(ir.l, 0), mul) for ir, mul in self.irrep_groups]
        )

        return eliminated.simplified() if simplified else eliminated
    
    def __mul__(self, other: Union[str, Tuple, List, Irrep, 'Irreps']) -> 'Irreps':
        """
        Compute the tensor product between two sets of irreducible representations.

        Args:
            other (Irreps or convertible): The other representation(s) to multiply with. 
                Automatically converted to ``Irreps`` if not already (see class constructor for valid formats).

        Returns:
            Irreps: Simplified direct sum containing all possible couplings between
            individual components of ``self`` and ``other``.

        Raises:
            ValueError: If input cannot be converted to valid Irreps.

        Examples:
            >>> Irreps("1x0e + 1x1o") * Irreps("1x0o + 1x1e")
            Irreps(2x0o+2x1e+1o+2o)
            # 0e * 0o -> 0o
            # 0e * 1e -> 1e
            # 1o * 0o -> 1e
            # 1o * 1e -> 0o+1o+2o 
            >>> Irreps("2x0e + 3x1o") * Irreps("1x1e + 2x2o")
            Irreps(3x0o+8x1e+3x1o+6x2e+7x2o+6x3e)
            # 2x0e * 1x1e -> 2x1e
            # 2x0e * 2x2o -> 4x2o
            # 3x1o * 1x1e -> 3x0o+3x1o+3x2o
            # 3x1o * 2x2o -> 6x1e+6x2e+6x3e

            >>> Irreps((0,2)) * Irreps((0,2))
            Irreps(3x0+6x1+6x2+3x3+4)
            >>> Irreps("1o") * Irreps("1o") * Irreps("1o")
            Irreps(0o+3x1o+2x2o+3o)
            >>> Irreps("1o") * Irreps("1o") * Irreps("1o") * Irreps("1")
            Irreps(3x0+6x1+6x2+3x3+4)
        
        Notes:
            Each component pair :math:`(l_1,p_1)`, :math:`(l_2,p_2)` produces irreps with:
            - :math:`l` in :math:`[|l_1-l_2|, ..., l_1+l_2]`
            - :math:`p = p_1 p_2` (parity product), None parity is treated as 0
            - Multiplicity = :math:`m_1 m_2`
            
            Final result combines all pairs and merges equivalent :math:`(l,p)` terms.
        """
        other = check_irreps(other)
        return sum((
            (ir, mul1 * mul2) 
            for ir1, mul1 in self.irrep_groups 
            for ir2, mul2 in other.irrep_groups
            for ir in ir1 * ir2
        ), start=Irreps("")).simplified()
    
    def __iter__(self):
        """
        Iterate through individual irreps with multiplicities expanded.

        Yields:
            Irrep: Individual irreducible representations repeated according to their multiplicities.

        Examples:
            >>> list(Irreps("2x0+2x0+2x0e+3x1o+2"))
            [0, 0, 0, 0, 0e, 0e, 1o, 1o, 1o, 2]
        """
        for ir, mul in self.irrep_groups:
            for _ in range(mul):
                yield ir

    def __len__(self) -> int:
        """
        Return total number of irreps including multiplicities.

        Returns:
            int: Sum of all multiplicities in the representation.

        Examples:
            >>> len(Irreps("2x0e + 3x1o"))
            5
        """
        return sum(mul for ir, mul in self.irrep_groups)

    def __getitem__(self, index: Union[int, slice]) -> Union[Irrep, 'Irreps']:
        """
        Access individual irrep(s) by index or slice (with step=1) in the expanded multiplicity sequence.

        Args:
            index (int or slice): Index or slice (step must be 1) in the flattened sequence of irreps.

        Returns:
            Irrep or Irreps: The irrep at the specified index, or a new Irreps object 
            containing the irreps selected by the slice.

        Raises:
            IndexError: If index is out of bounds.
            ValueError: If slice step is not 1.
            TypeError: If index is not an int or slice.

        Examples:
            >>> irreps = Irreps("2x0e + 1x1o + 2x2")
            >>> irreps[0]
            0e
            >>> irreps[2]
            1o
            >>> irreps[3]
            2
            >>> irreps[1:4]
            Irreps(0e+1o+2)
            >>> irreps[:2]
            Irreps(0e+0e)
            >>> irreps[3:]
            Irreps(2+2)
            >>> irreps[::2]  # Raises ValueError
            Traceback (most recent call last):
                ...
            ValueError: Irreps slicing only supports a step of 1
        """
        if isinstance(index, int):
            # Handle negative indices
            if index < 0:
                index += len(self)
            if not (0 <= index < len(self)):
                raise IndexError("Irrep index out of range")

            # Find the irrep at the specific index
            acc_mul = 0
            for ir, mul in self.irrep_groups:
                acc_mul += mul
                if acc_mul > index:
                    return ir
            # This part should theoretically not be reached if len(self) is correct
            # and index is within bounds, but added for robustness.
            raise IndexError("Irrep index out of range (internal error)")

        elif isinstance(index, slice):
            # Get the range parameters and check step
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Irreps slicing only supports a step of 1")

            # Collect irreps for the slice
            sliced_irreps = []
            current_irrep_idx = 0
            acc_mul = 0
            irrep_iter = iter(self.irrep_groups)
            # Initialize ir, mul before the loop
            try:
                ir, mul = next(irrep_iter)
            except StopIteration: # Handle empty Irreps case
                 ir, mul = None, 0

            for i in range(start, stop, step): # step is guaranteed to be 1 here
                # Advance irrep group if needed
                while ir is not None and not (acc_mul <= i < acc_mul + mul):
                    acc_mul += mul
                    try:
                        ir, mul = next(irrep_iter)
                    except StopIteration:
                         ir, mul = None, 0 # End of irrep groups

                if ir is None: # Should not happen if indices are correct
                     raise IndexError("Slice index out of range during iteration")
                
                sliced_irreps.append(ir)
                
            # Return a new Irreps object
            # Using the list constructor handles the [(Irrep, 1), ...] format implicitly
            return Irreps(sliced_irreps)
        else:
            raise TypeError(f"Irreps indices must be integers or slices, not {type(index).__name__}")

    def short_repr(self):
        return show_irreps(self.irrep_groups)
    
    def isomorphic_to(self, other):
        """
        Returns if self and other are two isomorphic Irreps, that is, 
        they are equals after simplification.

        Parameters
        ----------
        other : Irreps or convertible
            Representation to compare against

        Returns
        -------
        bool
            True if both representations have identical components after simplification

        Examples
        --------
        >>> Irreps("0e+0e+1o").isomorphic_to("1x1o+2x0e")
        True
        """
        other = check_irreps(other)
        return self.simplified() == other.simplified()
    @property
    def l_max(self):
        return max(ir.l for ir, _ in self.irrep_groups)
    
    @property
    def dim(self):
        """
        Total dimensionality of the representation space.

        Calculated as sum of (irrep dimension × multiplicity) for all components

        Examples
        --------
        >>> Irreps("2x0e + 3x1o").dim  # 2 * 1 + 3 * 3
        11
        """
        return sum(ir.dim * mul for ir, mul in self.irrep_groups)
    
    def dims(self):
        """
        Generator yielding dimensionality of each irrep in expanded sequence.

        Yields
        ------
        int
            Dimension of each individual irrep instance

        Examples
        --------
        >>> list(Irreps("2x0+2x0e+3x1o+2").dims())
        [1, 1, 1, 1, 3, 3, 3, 5]
        """
        for ir in self:
            yield ir.dim

    def __hash__(self):
        return self.short_repr().__hash__()
    
    def channels_divided(self, mul: int):
        """
        Divide all multiplicities by an integer factor.

        Parameters
        ----------
        mul : int
            The integer factor to divide all multiplicities by. Must exactly divide
            all current multiplicities.

        Returns
        -------
        Irreps
            New Irreps with all multiplicities divided by mul

        Raises
        ------
        AssertionError
            If any multiplicity is not divisible by mul

        Examples
        --------
        >>> Irreps("4x0e + 6x1o").channels_divided(2)
        Irreps(2x0e+3x1o)
        """
        assert all(multiplicities%mul==0 for ir, multiplicities in self.irrep_groups), (
            f"All multiplicities must be divisible by {mul}, but got {self.irrep_groups} "
            f"which contains multiplicities not divisible by {mul}"
        )
        return Irreps([(ir, multiplicities//mul) for ir, multiplicities in self.irrep_groups])
    
    
    def channels_multiplied(self, mul: int):
        """
        Multiply all multiplicities by an integer factor.

        Parameters
        ----------
        mul : int
            The integer factor to multiply all multiplicities by

        Returns
        -------
        Irreps
            New Irreps with all multiplicities multiplied by mul

        Examples
        --------
        >>> Irreps("2x0e + 3x1o").channels_multiplied(2)
        Irreps(4x0e+6x1o)
        """
        return Irreps([(ir, multiplicities*mul) for ir, multiplicities in self.irrep_groups])

    def max_channels(self):
        """
        Compute the greatest common divisor (GCD) of all multiplicities in the Irreps.

        This represents the largest integer factor by which all multiplicities can be divided
        while remaining integers.

        Returns
        -------
        int
            The GCD of all multiplicities in the representation

        Examples
        --------
        >>> Irreps("128x0e + 64x1o + 32x2e").max_channels()
        32
        >>> Irreps("3x0 + 5x1 + 7x2").max_channels()
        1
        """
        return math.gcd(*(multiplicities for ir, multiplicities in self.irrep_groups))

    def squeeze_max_channels(self):
        """
        Simplify the representation by dividing all multiplicities by their GCD and return
        both the simplified representation and the GCD factor.

        Returns
        -------
        tuple[Irreps, int]
            A tuple containing:
            - The simplified Irreps with multiplicities divided by their GCD
            - The max channels that was divided out

        Examples
        --------
        >>> Irreps("128x0e + 64x1o + 32x2e").squeeze_max_channels()
        (Irreps(4x0e+2x1o+2e), 32)
        >>> Irreps("3x0 + 5x1 + 7x2").squeeze_max_channels()
        (Irreps(3x0+5x1+7x2), 1)

        Notes
        -----
        This is useful when you need both the simplified representation and the scaling
        factor that was removed from the multiplicities.
        """
        max_channels = self.max_channels()
        return self.channels_divided(max_channels), max_channels


def unique_irreps(irreps):
    r"""
    Identify unique irreducible representations (irreps) in a given Irreps object and
    provide a mapping from each unique irrep to the indices of its occurrences in the
    original Irreps object.

    Parameters
    ----------
    irreps : Irreps
        An Irreps object containing a collection of irreducible representations.

    Returns
    -------
    tuple[Irreps, dict[Irrep, list[int]]]
        A tuple containing:
        - uniqued_irreps (Irreps): An Irreps object containing only the unique irreps
          from the input, maintaining the order of first appearance.
        - unique_mapping (dict[Irrep, list[int]]): A dictionary where keys are the
          unique Irrep objects and values are lists of indices indicating where
          each unique irrep appeared in the original ``irreps`` sequence.

    Examples
    --------
    >>> irreps_obj = Irreps("0+0e+1o+0e+0e")
    >>> uniqued, mapping = unique_irreps(irreps_obj)
    >>> print(uniqued)
    Irreps(0+0e+1o)
    >>> print(mapping)
    {0: [0], 0e: [1, 3, 4], 1o: [2]}

    >>> irreps_obj_2 = Irreps("2x1e+0o+1e")
    >>> uniqued_2, mapping_2 = unique_irreps(irreps_obj_2)
    >>> print(uniqued_2)
    Irreps(1e+0o)
    >>> print(mapping_2)
    {1e: [0, 1, 3], 0o: [2]}
    """
    uniqued_list = []
    unique_mapping = {}
    for i, ir in enumerate(irreps):
        if ir not in unique_mapping:
            uniqued_list.append(ir)
            unique_mapping[ir] = [i]
        else:
            unique_mapping[ir].append(i)
    uniqued_irreps = Irreps(uniqued_list)
    return uniqued_irreps, unique_mapping

def irrep_segments(irreps):
    r"""
    Compute segment boundaries for the irreps.
    
    Parameters
    ----------
    irreps : Irreps-like
        Input irreps, can be any object that can be parsed to an ``Irreps``.
    
    Returns
    -------
    torch.Tensor
        1D tensor of integers where the i-th element represents the starting index of the i-th irrep's block.
        The last element is the total dimension of all irreps combined.
    
    Examples
    --------
    >>> irrep_segments(Irreps("1o"))
    [0, 3]
    >>> irrep_segments(Irreps((0,4)))
    [0, 1, 4, 9, 16, 25]
    >>> irrep_segments("0+2x1e+3x2e+4")
    [0, 1, 4, 7, 12, 17, 22, 31]
    """
    irreps = check_irreps(irreps)
    acc_dim = 0
    seg = []
    for irrep in irreps:
        seg.append(acc_dim)
        acc_dim += irrep.dim
    seg.append(acc_dim)

    return seg

def irrep_indices(irreps):
    r"""
    Generate the mapping of the tensor indices to their corresponding irrep indices.
    
    Parameters
    ----------
    irreps : Irreps-like
        Input irreps, can be any object that can be parsed to an ``Irreps``.
    
    Returns
    -------
    torch.Tensor
        1D tensor where each element indicates the index of the irrep (in the input list)
        that the corresponding element belongs to.
    
    Examples
    --------
    >>> irrep_indices(Irreps("1o"))
    tensor([0, 0, 0])
    >>> irrep_indices(Irreps((0,3)))
    tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
    >>> irrep_indices("0+2x1e+2e")
    tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    """
    irreps = check_irreps(irreps)
    ind = []
    for i, irrep_obj in enumerate(irreps):
        ind.extend([i] * irrep_obj.dim)

    return torch.tensor(ind, dtype=torch.long)

def irrep_degrees(irreps):
    irreps = check_irreps(irreps)
    degrees = []
    for irrep_obj in irreps:
        degrees.append(irrep_obj.l)
    return torch.tensor(degrees, dtype=torch.long)

def element_degrees(irreps):
    irreps = check_irreps(irreps)
    degrees = []
    for irrep_obj in irreps:
        degrees.extend([irrep_obj.l] * irrep_obj.dim)
    return torch.tensor(degrees, dtype=torch.long)

def element_orders(irreps):
    irreps = check_irreps(irreps)
    orders = []
    for irrep_obj in irreps:
        for m_val in range(-irrep_obj.l, irrep_obj.l + 1):
            orders.append(m_val)
    return torch.tensor(orders, dtype=torch.long)
