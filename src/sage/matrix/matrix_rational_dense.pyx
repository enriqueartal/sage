"""
Dense matrices over the rational field

EXAMPLES:

We create a 3x3 matrix with rational entries and do some
operations with it.

::

    sage: a = matrix(QQ, 3,3, [1,2/3, -4/5, 1,1,1, 8,2, -3/19]); a
    [    1   2/3  -4/5]
    [    1     1     1]
    [    8     2 -3/19]
    sage: a.det()
    2303/285
    sage: a.charpoly()
    x^3 - 35/19*x^2 + 1259/285*x - 2303/285
    sage: b = a^(-1); b
    [ -615/2303  -426/2303   418/2303]
    [ 2325/2303  1779/2303  -513/2303]
    [-1710/2303   950/2303    95/2303]
    sage: b.det()
    285/2303
    sage: a == b
    False
    sage: a < b
    False
    sage: b < a
    True
    sage: a > b
    True
    sage: a*b
    [1 0 0]
    [0 1 0]
    [0 0 1]

TESTS::

    sage: a = matrix(QQ, 2, range(4), sparse=False)
    sage: TestSuite(a).run()
"""

#*****************************************************************************
#       Copyright (C) 2004,2005,2006 William Stein <wstein@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from __future__ import absolute_import, print_function

from libc.string cimport strcpy, strlen


from sage.modules.vector_rational_dense cimport Vector_rational_dense
from sage.ext.stdsage cimport PY_NEW
from sage.misc.randstate cimport randstate, current_randstate

from sage.modules.vector_rational_dense cimport Vector_rational_dense

from cysignals.signals cimport sig_on, sig_off, sig_check
from cysignals.memory cimport sig_malloc, sig_free

from sage.arith.rational_reconstruction cimport mpq_rational_reconstruction

from sage.libs.gmp.types cimport mpz_t, mpq_t
from sage.libs.gmp.mpz cimport mpz_init, mpz_clear, mpz_cmp_si
from sage.libs.gmp.mpq cimport mpq_init, mpq_clear, mpq_set_si, mpq_mul, mpq_add, mpq_set
from sage.libs.gmp.randomize cimport (mpq_randomize_entry, mpq_randomize_entry_as_int, mpq_randomize_entry_recip_uniform,
    mpq_randomize_entry_nonzero, mpq_randomize_entry_as_int_nonzero, mpq_randomize_entry_recip_uniform_nonzero)

from sage.libs.flint.fmpz cimport *
from sage.libs.flint.fmpq cimport *
from sage.libs.flint.fmpz_mat cimport *
from sage.libs.flint.fmpq_mat cimport *

cimport sage.structure.element

from sage.structure.sequence import Sequence
from sage.rings.rational cimport Rational
from .matrix cimport Matrix
from .matrix_integer_dense cimport Matrix_integer_dense, _lift_crt
from sage.structure.element cimport ModuleElement, RingElement, Element, Vector
from sage.rings.integer cimport Integer
from sage.rings.ring import is_Ring
from sage.rings.integer_ring import ZZ, is_IntegerRing
from sage.rings.finite_rings.finite_field_constructor import FiniteField as GF
from sage.rings.finite_rings.integer_mod_ring import is_IntegerModRing
from sage.rings.rational_field import QQ
from sage.arith.all import gcd

from .matrix2 import cmp_pivots, decomp_seq
from .matrix0 import Matrix as Matrix_base

from sage.misc.all import verbose, get_verbose, prod

#########################################################
# PARI C library
from cypari2.gen cimport Gen
from sage.libs.pari.convert_gmp cimport INTFRAC_to_mpq
from sage.libs.pari.convert_flint cimport rational_matrix, _new_GEN_from_fmpq_mat_t
from cypari2.stack cimport clear_stack
from cypari2.paridecl cimport *
#########################################################

cdef class Matrix_rational_dense(Matrix_dense):

    ########################################################################
    # LEVEL 1 functionality
    # x * __cinit__
    # x * __dealloc__
    # x * __init__
    # x * set_unsafe
    # x * get_unsafe
    # x * cdef _pickle
    # x * cdef _unpickle
    ########################################################################
    def __cinit__(self, parent, entries, copy, coerce):
        """
        Create and allocate memory for the matrix.

        EXAMPLES::

            sage: from sage.matrix.matrix_rational_dense import Matrix_rational_dense
            sage: a = Matrix_rational_dense.__new__(Matrix_rational_dense, Mat(ZZ,3), 0,0,0)
            sage: type(a)
            <type 'sage.matrix.matrix_rational_dense.Matrix_rational_dense'>

        .. warning::

           This is for internal use only, or if you really know what
           you're doing.
        """
        Matrix_dense.__init__(self, parent)

        sig_on()
        fmpq_mat_init(self._matrix, self._nrows, self._ncols)
        sig_off()

    def  __dealloc__(self):
        sig_on()
        fmpq_mat_clear(self._matrix)
        sig_off()

    def __init__(self, parent, entries=None, coerce=True, copy=True):
        r"""
        TESTS::

            sage: matrix(QQ, 2, 2, 1/4)
            [1/4   0]
            [  0 1/4]
            sage: matrix(QQ, 3, 1, [1/2, -3/4, 0])
            [ 1/2]
            [-3/4]
            [   0]
        """
        cdef Py_ssize_t i, j, k
        cdef Rational z

        if entries is None: return
        if isinstance(entries, xrange):
            entries = list(entries)
        if isinstance(entries, (list, tuple)):
            if len(entries) != self._nrows * self._ncols:
                raise TypeError("entries has the wrong length")

            if coerce:
                k = 0
                for i in range(self._nrows):
                    for j in range(self._ncols):
                    # TODO: Should use an unsafe un-bounds-checked array access here.
                        sig_check()
                        z = Rational(entries[k])
                        k += 1
                        fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), z.value)
            else:
                k = 0
                for i in range(self._nrows):
                    for j in range(self._ncols):
                    # TODO: Should use an unsafe un-bounds-checked array access here.
                        sig_check()
                        fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), (<Rational> entries[k]).value)
                        k += 1

        else:
            # is it a scalar?
            try:
                # Try to coerce entries to a scalar (an integer)
                z = Rational(entries)
                is_list = False
            except TypeError:
                raise TypeError("entries must be coercible to a list or integer")

            if z:
                if self._nrows != self._ncols:
                    raise TypeError("nonzero scalar matrix must be square")
                for i in range(self._nrows):
                    fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, i), z.value)

    cdef set_unsafe(self, Py_ssize_t i, Py_ssize_t j, value):
        fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), (<Rational> value).value)

    cdef get_unsafe(self, Py_ssize_t i, Py_ssize_t j):
        cdef Rational x
        x = Rational.__new__(Rational)
        fmpq_get_mpq(x.value, fmpq_mat_entry(self._matrix, i, j))
        return x

    cdef _add_ui_unsafe_assuming_int(self, Py_ssize_t i, Py_ssize_t j, unsigned long int n):
        # doesn't check immutability
        # doesn't do bounds checks.
        # assumes that self[i,j] is an integer.
        cdef fmpz * entry = fmpq_numref(fmpq_mat_entry(self._matrix, i, j))
        fmpz_add_ui(entry, entry, n)

    cdef _sub_ui_unsafe_assuming_int(self, Py_ssize_t i, Py_ssize_t j, unsigned long int n):
        # doesn't check immutability
        # doesn't do bounds checks.
        # assumes that self[i,j] is an integer.
        cdef fmpz * entry = fmpq_numref(fmpq_mat_entry(self._matrix, i, j))
        fmpz_sub_ui(entry, entry, n)

    def _pickle(self):
        return self._pickle_version0(), 0

    def _unpickle(self, data, int version):
        if version == 0:
            self._unpickle_version0(data)
        else:
            raise RuntimeError("unknown matrix version (=%s)"%version)

    cdef _pickle_version0(self):
        return self._export_as_string(32)

    cpdef _export_as_string(self, int base=10):
        """
        Return space separated string of the entries in this matrix, in the
        given base. This is optimized for speed.

        INPUT: base -an integer = 36; (default: 10)

        EXAMPLES::

            sage: m = matrix(QQ,2,3,[1,2/3,-3/4,1,-2/3,-45/17])
            sage: m._export_as_string(10)
            '1 2/3 -3/4 1 -2/3 -45/17'
            sage: m._export_as_string(16)
            '1 2/3 -3/4 1 -2/3 -2d/11'

        """
        cdef Py_ssize_t i, j, len_so_far, m, n
        cdef char *a
        cdef char *s
        cdef char *t
        cdef char *tmp

        if self._nrows == 0 or self._ncols == 0:
            data = ''
        else:
            n = self._nrows * self._ncols * 10
            s = <char*> sig_malloc(n * sizeof(char))
            t = s
            len_so_far = 0

            sig_on()
            for i in range(self._nrows):
                for j in range(self._ncols):
                    m = fmpz_sizeinbase (fmpq_mat_entry_num(self._matrix, i, j), base) + \
                        fmpz_sizeinbase (fmpq_mat_entry_den(self._matrix, i, j), base) + 3
                    if len_so_far + m + 1 >= n:
                        # copy to new string with double the size
                        n = 2*n + m + 1
                        tmp = <char*> sig_malloc(n * sizeof(char))
                        strcpy(tmp, s)
                        sig_free(s)
                        s = tmp
                        t = s + len_so_far
                    fmpq_get_str(t, base, fmpq_mat_entry(self._matrix, i, j))
                    m = strlen(t)
                    len_so_far = len_so_far + m + 1
                    t = t + m
                    t[0] = <char>32
                    t[1] = <char>0
                    t = t + 1
            sig_off()
            data = str(s)[:-1]
            sig_free(s)
        return data

    cdef _unpickle_version0(self, data):
        r"""
        TESTS::

            sage: a = random_matrix(QQ, 4, 3, num_bound=2**500, den_bound=2**500)
            sage: loads(dumps(a)) == a  # indirect doctest
            True
        """
        cdef Py_ssize_t i, j, k
        data = data.split()
        if len(data) != self._nrows * self._ncols:
            raise RuntimeError("invalid pickle data")
        k = 0
        for i in range(self._nrows):
            for j in range(self._ncols):
                s = data[k]
                k += 1
                if '/' in s:
                    num, den = s.split('/')
                    if fmpz_set_str(fmpq_mat_entry_num(self._matrix, i, j), num, 32) or \
                       fmpz_set_str(fmpq_mat_entry_den(self._matrix, i, j), den, 32):
                           raise RuntimeError("invalid pickle data")
                else:
                    if fmpz_set_str(fmpq_mat_entry_num(self._matrix, i, j), s, 32):
                        raise RuntimeError("invalid pickle data")
                    fmpz_one(fmpq_mat_entry_den(self._matrix, i, j))

    def __hash__(self):
        return self._hash()

    ########################################################################
    # LEVEL 2 functionality
    # x * cdef _add_
    # x * cdef _mul_
    # x * cdef _vector_times_matrix_
    # x * cpdef _cmp_
    # x * __neg__
    #   * __invert__
    # x * __copy__
    # x * _multiply_classical
    #   * _list -- list of underlying elements (need not be a copy)
    #   * _dict -- sparse dictionary of underlying elements (need not be a copy)
    ########################################################################

    cpdef _lmul_(self, RingElement right):
        """
        EXAMPLES::

            sage: a = matrix(QQ, 2, range(6))
            sage: (3/4) * a
            [   0  3/4  3/2]
            [ 9/4    3 15/4]
        """
        cdef Matrix_rational_dense M
        cdef fmpq_t x
        fmpq_init(x)
        fmpq_set_mpq(x, (<Rational>right).value)
        M = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)
        fmpq_mat_scalar_mul_fmpz(M._matrix, self._matrix, fmpq_numref(x))
        fmpq_mat_scalar_div_fmpz(M._matrix, M._matrix, fmpq_denref(x))
        fmpq_clear(x)
        return M

    cpdef _add_(self, right):
        """
        Add two dense matrices over QQ.

        EXAMPLES::

            sage: a = MatrixSpace(QQ,3)(range(9))
            sage: b = MatrixSpace(QQ,3)([1/n for n in range(1,10)])
            sage: a+b
            [   1  3/2  7/3]
            [13/4 21/5 31/6]
            [43/7 57/8 73/9]
            sage: b.swap_rows(1,2)
            sage: #a+b
        """
        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)

        sig_on()
        fmpq_mat_add(ans._matrix, self._matrix, (<Matrix_rational_dense> right)._matrix)
        sig_off()
        return ans

    cpdef _sub_(self, right):
        """
        Subtract two dense matrices over QQ.

        EXAMPLES::

            sage: a = MatrixSpace(QQ,3)(range(9))
            sage: b = MatrixSpace(QQ,3)([1/n for n in range(1,10)])
            sage: a-b
            [  -1  1/2  5/3]
            [11/4 19/5 29/6]
            [41/7 55/8 71/9]
        """
        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)

        sig_on()
        fmpq_mat_sub(ans._matrix, self._matrix, (<Matrix_rational_dense> right)._matrix)
        sig_off()
        return ans

    cpdef int _cmp_(self, right) except -2:
        r"""
        TESTS::

            sage: M = MatrixSpace(QQ, 1)
            sage: M(1) < M(2)
            True
            sage: M(1/3) >= M(5/2)
            False
            sage: M(2) == M(2)
            True
            sage: M(3/4) != M(2)
            True
        """
        cdef Py_ssize_t i, j
        cdef int k
        for i in range(self._nrows):
            for j in range(self._ncols):
                k = fmpq_cmp(fmpq_mat_entry(self._matrix, i, j),
                             fmpq_mat_entry((<Matrix_rational_dense> right)._matrix, i, j))
                if k:
                    return (k > 0) - (k < 0)
        return 0

    cdef _vector_times_matrix_(self, Vector v):
        """
        Returns the vector times matrix product.

        INPUT:


        -  ``v`` - a free module element.


        OUTPUT: The vector times matrix product v\*A.

        EXAMPLES::

            sage: B = matrix(QQ,2, [1,2,3,4])
            sage: V = QQ^2
            sage: w = V([-1,5/2])
            sage: w*B
            (13/2, 8)
        """
        cdef Vector_rational_dense w, ans
        cdef Py_ssize_t i, j
        cdef mpq_t x, y, z

        M = self._row_ambient_module()
        w = <Vector_rational_dense> v
        ans = M.zero_vector()

        mpq_init(x)
        mpq_init(y)
        mpq_init(z)
        for i in range(self._ncols):
            mpq_set_si(x, 0, 1)
            for j in range(self._nrows):
                fmpq_get_mpq(z, fmpq_mat_entry(self._matrix, j, i))
                mpq_mul(y, w._entries[j], z)
                mpq_add(x, x, y)
            mpq_set(ans._entries[i], x)
        mpq_clear(x)
        mpq_clear(y)
        mpq_clear(z)
        return ans


    def __neg__(self):
        """
        Negate a matrix over QQ.

        EXAMPLES::

            sage: a = MatrixSpace(QQ,3)([1/n for n in range(1,10)])
            sage: -a
            [  -1 -1/2 -1/3]
            [-1/4 -1/5 -1/6]
            [-1/7 -1/8 -1/9]
        """
        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)
        fmpq_mat_neg(ans._matrix, self._matrix)
        return ans

    def __copy__(self):
        """
        Copy a matrix over QQ.

        EXAMPLES::

            sage: a = MatrixSpace(QQ,3)([1/n for n in range(1,10)])
            sage: -a
            [  -1 -1/2 -1/3]
            [-1/4 -1/5 -1/6]
            [-1/7 -1/8 -1/9]
        """
        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)
        fmpq_mat_set(ans._matrix, self._matrix)
        if self._subdivisions is not None:
            ans.subdivide(*self.subdivisions())
        return ans

    # cdef _mul_(self, Matrix right):
    # cpdef int _cmp_(self, Matrix right) except -2:
    # def __invert__(self):
    # def _list(self):
    # def _dict(self):


    ########################################################################
    # LEVEL 3 functionality (Optional)
    # x * cdef _sub_
    #   * __deepcopy__
    #   * __invert__
    #   * Matrix windows -- only if you need strassen for that base
    #   * Other functions (list them here):
    # x * denom(self):
    # x * mpz_denom(self, mpz_t d):
    # x * _clear_denom(self):
    # o * echelon_modular(self, height_guess=None):
    ########################################################################
    def __invert__(self):
        """
        EXAMPLES::

            sage: a = matrix(QQ,3,range(9))
            sage: a.inverse()
            Traceback (most recent call last):
            ...
            ZeroDivisionError: input matrix must be nonsingular
            sage: a = matrix(QQ, 2, [1, 5, 17, 3])
            sage: a.inverse()
            [-3/82  5/82]
            [17/82 -1/82]
        """
        return self.__invert__main()

    def _invert_flint(self):
        r"""
        TESTS::

            sage: matrix(QQ, 2, [1,2,3,4])._invert_flint()
            [  -2    1]
            [ 3/2 -1/2]
        """
        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, self._parent, None, None, None)
        sig_on()
        fmpq_mat_inv(ans._matrix, self._matrix)
        sig_off()
        return ans

    def __invert__main(self, check_invertible=True, algorithm=None):
        """
        Return the inverse of this matrix

        INPUT:


        -  ``check_invertible`` - default: True (whether to
           check that matrix is invertible)

        - ``algorithm`` -- (optional) one of ``'flint'``,  ``'pari'``, ``'iml'``

        EXAMPLES::

            sage: a = matrix(QQ,3,[1,2,5,3,2,1,1,1,1,])
            sage: a.__invert__main(check_invertible=False)
            [1/2 3/2  -4]
            [ -1  -2   7]
            [1/2 1/2  -2]

        A 1x1 matrix (a special case)::

            sage: a = matrix(QQ, 1, [390284089234])
            sage: a.__invert__main()
            [1/390284089234]

        A 2x2 matrix (a special hand-coded case)::

            sage: a = matrix(QQ, 2, [1, 5, 17, 3]); a
            [ 1  5]
            [17  3]
            sage: a.inverse()
            [-3/82  5/82]
            [17/82 -1/82]
            sage: a.__invert__main()  * a
            [1 0]
            [0 1]
        """
        if self._nrows != self._ncols:
            raise ArithmeticError("self must be a square matrix")

        if self._nrows == 0:
            return self

        if algorithm is None:
            if self._nrows <= 25 and self.height().ndigits() <= 100:
                algorithm = "pari"
            else:
                algorithm = "iml"

        if algorithm == "flint":
            return self._invert_flint()

        if algorithm == "pari":
            from sage.libs.pari.all import PariError
            try:
                return self._invert_pari()
            except PariError:
                # Assume the error is because the matrix is not invertible.
                raise ZeroDivisionError("input matrix must be nonsingular")

        if algorithm == "iml":
            AZ, denom = self._clear_denom()
            B, d = AZ._invert_iml(check_invertible=check_invertible)
            return (denom/d)*B


        else:
            raise ValueError("unknown algorithm '%s'"%algorithm)

    def determinant(self, algorithm=None, proof=None):
        """
        Return the determinant of this matrix.

        INPUT:

        -  ``proof`` - bool or None; if None use
           proof.linear_algebra(); only relevant for the padic algorithm.

        - ``algorithm`` -- (optional) one of ``'pari'``, ``'integer'``

        .. NOTE::

           It would be *VERY VERY* hard for det to fail even with
           proof=False.


        ALGORITHM: Clear denominators and call the integer determinant
        function.

        EXAMPLES::

            sage: m = matrix(QQ,3,[1,2/3,4/5, 2,2,2, 5,3,2/5])
            sage: m.determinant()
            -34/15
            sage: m.charpoly()
            x^3 - 17/5*x^2 - 122/15*x + 34/15
        """
        det = self.fetch('det')
        if det is not None:
            return det

        if algorithm is None:
            if self._nrows <= 7:
                algorithm = "pari"
            else:
                algorithm = "integer"

        if algorithm == "pari":
            det = self._det_pari()
        elif algorithm == "flint":
            det = self._det_flint()
        elif algorithm == "integer":
            A, denom = self._clear_denom()
            det = Rational(A.determinant(proof=proof))
            if not denom.is_one():
                det = det / (denom ** self.nrows())
        else:
            raise ValueError("unknown algorithm '%s'"%algorithm)

        self.cache('det', det)
        return det

    def _det_flint(self):
        r"""
        Return the determinant (computed using flint)

        EXAMPLES::

            sage: matrix(QQ, 2, [1/3, 2/5, 3/4, 7/8])._det_flint()
            -1/120
            sage: matrix(QQ, 0)._det_flint()
            1
            sage: matrix(QQ, 1, [0])._det_flint()
            0
        """
        cdef Rational d = Rational.__new__(Rational)
        cdef fmpq_t e
        fmpq_init(e)
        sig_on()
        fmpq_mat_det(e, self._matrix)
        fmpq_get_mpq(d.value, e)
        sig_off()
        return d

    def denominator(self):
        """
        Return the denominator of this matrix.

        OUTPUT: a Sage Integer

        EXAMPLES::

            sage: b = matrix(QQ,2,range(6)); b[0,0]=-5007/293; b
            [-5007/293         1         2]
            [        3         4         5]
            sage: b.denominator()
            293
        """
        cdef Integer z = Integer.__new__(Integer)
        cdef fmpz_t tmp
        fmpz_init(tmp)
        self.fmpz_denom(tmp)
        fmpz_get_mpz(z.value, tmp)
        fmpz_clear(tmp)
        return z

    cdef int fmpz_denom(self, fmpz_t d) except -1:
        cdef Py_ssize_t i, j
        sig_on()
        fmpz_one(d)
        for i in range(self._nrows):
            for j in range(self._ncols):
                fmpz_lcm(d, d, fmpq_mat_entry_den(self._matrix, i, j))
        sig_off()
        return 0

    def _clear_denom(self):
        """
        INPUT:


        -  ``self`` - a matrix


        OUTPUT: D\*self, D

        The product is a matrix over ZZ

        EXAMPLES::

            sage: a = matrix(QQ,2,[-1/6,-7,3,5/4]); a
            [-1/6   -7]
            [   3  5/4]
            sage: a._clear_denom()
            (
            [ -2 -84]
            [ 36  15], 12
            )
        """
        X = self.fetch('clear_denom')
        if X is not None:
            return X

        cdef Py_ssize_t i, j
        cdef Matrix_integer_dense A
        cdef fmpz * tmp
        cdef fmpz_t denom
        fmpz_init(denom)
        self.fmpz_denom(denom)

        from sage.matrix.matrix_space import MatrixSpace
        MZ = MatrixSpace(ZZ, self._nrows, self._ncols, sparse=False)
        A =  Matrix_integer_dense.__new__(Matrix_integer_dense, MZ, None, None, None)

        sig_on()
        for i in range(self._nrows):
            for j in range(self._ncols):
                tmp = fmpz_mat_entry(A._matrix, i, j)
                fmpz_divexact(tmp, denom, fmpq_mat_entry_den(self._matrix, i, j))
                fmpz_mul(tmp, tmp, fmpq_mat_entry_num(self._matrix, i, j))
        sig_off()

        cdef Integer D = PY_NEW(Integer)
        fmpz_get_mpz(D.value, denom)
        fmpz_clear(denom)
        X = (A, D)
        self.cache('clear_denom', X)
        return X

    def charpoly(self, var='x', algorithm='linbox'):
        """
        Return the characteristic polynomial of this matrix.

        INPUT:


        -  ``var`` - 'x' (string)

        -  ``algorithm`` - 'linbox' (default) or 'generic'


        OUTPUT: a polynomial over the rational numbers.

        EXAMPLES::

            sage: a = matrix(QQ, 3, [4/3, 2/5, 1/5, 4, -3/2, 0, 0, -2/3, 3/4])
            sage: f = a.charpoly(); f
            x^3 - 7/12*x^2 - 149/40*x + 97/30
            sage: f(a)
            [0 0 0]
            [0 0 0]
            [0 0 0]

        TESTS:

        The cached polynomial should be independent of the ``var``
        argument (:trac:`12292`). We check (indirectly) that the
        second call uses the cached value by noting that its result is
        not cached::

            sage: M = MatrixSpace(QQ, 2)
            sage: A = M(range(0, 2^2))
            sage: type(A)
            <type 'sage.matrix.matrix_rational_dense.Matrix_rational_dense'>
            sage: A.charpoly('x')
            x^2 - 3*x - 2
            sage: A.charpoly('y')
            y^2 - 3*y - 2
            sage: A._cache['charpoly_linbox']
            x^2 - 3*x - 2

        """
        cache_key = 'charpoly_%s' % algorithm
        g = self.fetch(cache_key)
        if g is not None:
            return g.change_variable_name(var)

        if algorithm == 'linbox':
            A, denom = self._clear_denom()
            f = A.charpoly(var, algorithm='linbox')
            x = f.parent().gen()
            g = f(x * denom) * (1 / (denom**f.degree()))
        elif algorithm == 'generic':
            g = Matrix_dense.charpoly(self, var)
        else:
            raise ValueError("no algorithm '%s'"%algorithm)

        self.cache(cache_key, g)
        return g

    def minpoly(self, var='x', algorithm='linbox'):
        """
        Return the minimal polynomial of this matrix.

        INPUT:


        -  ``var`` - 'x' (string)

        -  ``algorithm`` - 'linbox' (default) or 'generic'


        OUTPUT: a polynomial over the rational numbers.

        EXAMPLES::

            sage: a = matrix(QQ, 3, [4/3, 2/5, 1/5, 4, -3/2, 0, 0, -2/3, 3/4])
            sage: f = a.minpoly(); f
            x^3 - 7/12*x^2 - 149/40*x + 97/30
            sage: a = Mat(ZZ,4)(range(16))
            sage: f = a.minpoly(); f.factor()
            x * (x^2 - 30*x - 80)
            sage: f(a) == 0
            True

        ::

            sage: a = matrix(QQ, 4, [1..4^2])
            sage: factor(a.minpoly())
            x * (x^2 - 34*x - 80)
            sage: factor(a.minpoly('y'))
            y * (y^2 - 34*y - 80)
            sage: factor(a.charpoly())
            x^2 * (x^2 - 34*x - 80)
            sage: b = matrix(QQ, 4, [-1, 2, 2, 0, 0, 4, 2, 2, 0, 0, -1, -2, 0, -4, 0, 4])
            sage: a = matrix(QQ, 4, [1, 1, 0,0, 0,1,0,0, 0,0,5,0, 0,0,0,5])
            sage: c = b^(-1)*a*b
            sage: factor(c.minpoly())
            (x - 5) * (x - 1)^2
            sage: factor(c.charpoly())
            (x - 5)^2 * (x - 1)^2
        """
        key = 'minpoly_%s_%s'%(algorithm, var)
        x = self.fetch(key)
        if x: return x

        if algorithm == 'linbox':
            A, denom = self._clear_denom()
            f = A.minpoly(var, algorithm='linbox')
            x = f.parent().gen()
            g = f(x * denom) * (1 / (denom**f.degree()))
        elif algorithm == 'generic':
            g = Matrix_dense.minpoly(self, var)
        else:
            raise ValueError("no algorithm '%s'"%algorithm)

        self.cache(key, g)
        return g

    cdef sage.structure.element.Matrix _matrix_times_matrix_(self, sage.structure.element.Matrix right):
        """
        EXAMPLES::

            sage: n = 3
            sage: a = matrix(QQ,n,range(n^2))/3
            sage: b = matrix(QQ,n,range(1, n^2 + 1))/5
            sage: a._multiply_classical(b)
            [ 6/5  7/5  8/5]
            [18/5 22/5 26/5]
            [   6 37/5 44/5]
        """
        if self._ncols != right._nrows:
            raise IndexError("Number of columns of self must equal number of rows of right.")

        if self._nrows == right._nrows:
            # self acts on the space of right
            parent = right.parent()
        if self._ncols == right._ncols:
            # right acts on the space of self
            parent = self.parent()
        else:
            parent = self.matrix_space(self._nrows, right._ncols)

        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, parent, None, None, None)

        sig_on()
        fmpq_mat_mul(ans._matrix, self._matrix, (<Matrix_rational_dense> right)._matrix)
        sig_off()
        return ans


# FIXME: what is the good strategy now?
#    cdef sage.structure.element.Matrix _matrix_times_matrix_(self, sage.structure.element.Matrix right):
#        """
#        Multiply matrices self and right, which are assumed to have
#        compatible dimensions and the same base ring.
#
#        Uses pari when all matrix dimensions are small (at most 6);
#        otherwise convert to matrices over the integers and multiply
#        those matrices.
#
#        EXAMPLES::
#
#            sage: a = matrix(3,[[1,2,3],[1/5,2/3,-1/3],[-4,5,6]])   # indirect test
#            sage: a*a
#            [ -53/5   55/3   61/3]
#            [   5/3 -37/45 -73/45]
#            [   -27   76/3   67/3]
#            sage: (pari(a)*pari(a)).sage() == a*a
#            True
#        """
#        if self._nrows <= 6 and self._ncols <= 6 and right._nrows <= 6 and right._ncols <= 6 and \
#               max(self.height().ndigits(),right.height().ndigits()) <= 10000:
#            return self._multiply_pari(right)
#        return self._multiply_over_integers(right)

    def _multiply_over_integers(self, Matrix_rational_dense right, algorithm='default'):
        """
        Multiply this matrix by right using a multimodular algorithm and
        return the result.

        INPUT:


        -  ``self`` - matrix over QQ

        -  ``right`` - matrix over QQ

        -  ``algorithm``

           - 'default': use whatever is the default for A\*B when A, B
             are over ZZ.

           - 'multimodular': use a multimodular algorithm


        EXAMPLES::

            sage: a = MatrixSpace(QQ,10,5)(range(50))
            sage: b = MatrixSpace(QQ,5,12)([1/n for n in range(1,61)])
            sage: a._multiply_over_integers(b) == a._multiply_over_integers(b, algorithm='multimodular')
            True

        ::

            sage: a = MatrixSpace(QQ,3)(range(9))
            sage: b = MatrixSpace(QQ,3)([1/n for n in range(1,10)])
            sage: a._multiply_over_integers(b, algorithm = 'multimodular')
            [ 15/28   9/20   7/18]
            [  33/7 117/40   20/9]
            [249/28   27/5  73/18]
        """
        cdef Matrix_integer_dense A, B, AB
        cdef Matrix_rational_dense res
        cdef Integer D
        sig_on()
        A, A_denom = self._clear_denom()
        B, B_denom = right._clear_denom()
        if algorithm == 'default' or algorithm == 'multimodular':
            AB = A*B
        else:
            sig_off()
            raise ValueError("unknown algorithm '%s'"%algorithm)
        D = A_denom * B_denom
        if self._nrows == right._nrows:
            # self acts on the space of right
            res = Matrix_rational_dense.__new__(Matrix_rational_dense, right.parent(), 0, 0, 0)
        if self._ncols == right._ncols:
            # right acts on the space of self
            res = Matrix_rational_dense.__new__(Matrix_rational_dense, self.parent(), 0, 0, 0)
        else:
            res = Matrix_rational_dense.__new__(Matrix_rational_dense, self.matrix_space(AB._nrows, AB._ncols), 0, 0, 0)
        for i in range(res._nrows):
            for j in range(res._ncols):
                fmpz_set(fmpq_mat_entry_num(res._matrix, i, j), fmpz_mat_entry(AB._matrix,i,j))
                fmpz_set_mpz(fmpq_mat_entry_den(res._matrix, i, j), D.value)
                fmpq_canonicalise(fmpq_mat_entry(res._matrix, i, j))
        sig_off()
        return res


    def height(self):
        """
        Return the height of this matrix, which is the maximum of the
        absolute values of all numerators and denominators of entries in
        this matrix.

        OUTPUT: an Integer

        EXAMPLES::

            sage: b = matrix(QQ,2,range(6)); b[0,0]=-5007/293; b
            [-5007/293         1         2]
            [        3         4         5]
            sage: b.height()
            5007
        """
        cdef Integer z
        cdef fmpz_t tmp
        fmpz_init(tmp)
        self.fmpz_height(tmp)
        z = PY_NEW(Integer)
        fmpz_get_mpz(z.value, tmp)
        fmpz_clear(tmp)
        return z

    cdef int fmpz_height(self, fmpz_t h) except -1:
        cdef fmpz_t x
        cdef int i, j
        sig_on()
        fmpz_init(x)
        fmpz_zero(h)
        for i in range(self._nrows):
            for j in range(self._ncols):
                fmpz_abs(x, fmpq_mat_entry_num(self._matrix, i, j))
                if fmpz_cmp(h, x) < 0:
                    fmpz_set(h, x)
                fmpz_abs(x, fmpq_mat_entry_den(self._matrix, i, j))
                if fmpz_cmp(h, x) < 0:
                    fmpz_set(h, x)
        fmpz_clear(x)
        sig_off()
        return 0

#    cdef int _rescale(self, mpq_t a) except -1:
#        cdef int i, j
#        sig_on()
#        for i from 0 <= i < self._nrows:
#            for j from 0 <= j < self._ncols:
#                mpq_mul(self._matrix[i][j], self._matrix[i][j], a)
#        sig_off()

    def _adjoint(self):
        """
        Return the adjoint of this matrix.

        Assumes self is a square matrix (checked in adjoint).

        EXAMPLES::

            sage: m = matrix(QQ,3,[1..9])/9; m
            [1/9 2/9 1/3]
            [4/9 5/9 2/3]
            [7/9 8/9   1]
            sage: m.adjoint()
            [-1/27  2/27 -1/27]
            [ 2/27 -4/27  2/27]
            [-1/27  2/27 -1/27]
        """
        return self.parent()(self.__pari__().matadjoint().sage())

    def _magma_init_(self, magma):
        """
        EXAMPLES::

            sage: m = matrix(QQ,2,3,[1,2/3,-3/4,1,-2/3,-45/17])
            sage: m._magma_init_(magma)
            'Matrix(RationalField(),2,3,StringToIntegerSequence("204 136 -153 204 -136 -540"))/204'
            sage: magma(m)                                                # optional - magma
            [     1    2/3   -3/4]
            [     1   -2/3 -45/17]
        """
        X, d = self._clear_denom()
        s = X._magma_init_(magma).replace('IntegerRing','RationalField')
        if d != 1:
            s += '/%s'%d._magma_init_(magma)
        return s

    def prod_of_row_sums(self, cols):
        cdef Py_ssize_t i, c
        cdef fmpq_t s, pr
        fmpq_init(s)
        fmpq_init(pr)

        fmpq_one(pr)
        for i in range(self._nrows):
            fmpq_zero(s)
            for c in cols:
                if c < 0 or c >= self._ncols:
                    raise IndexError("matrix column index out of range")
                fmpq_add(s, s, fmpq_mat_entry(self._matrix, i, c))
            fmpq_mul(pr, pr, s)
        cdef Rational ans
        ans = Rational.__new__(Rational)
        fmpq_get_mpq(ans.value, pr)
        fmpq_clear(s)
        fmpq_clear(pr)
        return ans

    def _right_kernel_matrix(self, **kwds):
        r"""
        Returns a pair that includes a matrix of basis vectors
        for the right kernel of ``self``.

        INPUT:

        - ``kwds`` - these are provided for consistency with other versions
          of this method.  Here they are ignored as there is no optional
          behavior available.

        OUTPUT:

        Returns a pair.  First item is the string 'computed-iml-rational'
        that identifies the nature of the basis vectors.

        Second item is a matrix whose rows are a basis for the right kernel,
        over the rationals, as computed by the IML library.  Notice that the
        IML library returns a matrix that is in the 'pivot' format, once the
        whole matrix is multiplied by -1.  So the 'computed' format is very
        close to the 'pivot' format.

        EXAMPLES::

            sage: A = matrix(QQ, [
            ....:                 [1, 0, 1, -3, 1],
            ....:                 [-5, 1, 0, 7, -3],
            ....:                 [0, -1, -4, 6, -2],
            ....:                 [4, -1, 0, -6, 2]])
            sage: result = A._right_kernel_matrix()
            sage: result[0]
            'computed-iml-rational'
            sage: result[1]
            [-1  2 -2 -1  0]
            [ 1  2  0  0 -1]
            sage: X = result[1].transpose()
            sage: A*X == zero_matrix(QQ, 4, 2)
            True

        Computed result is the negative of the pivot basis, which
        is just slightly more efficient to compute. ::

            sage: A.right_kernel_matrix(basis='pivot') == -A.right_kernel_matrix(basis='computed')
            True

        TESTS:

        We test three trivial cases. ::

            sage: A = matrix(QQ, 0, 2)
            sage: A._right_kernel_matrix()[1]
            [1 0]
            [0 1]
            sage: A = matrix(QQ, 2, 0)
            sage: A._right_kernel_matrix()[1].parent()
            Full MatrixSpace of 0 by 0 dense matrices over Rational Field
            sage: A = zero_matrix(QQ, 4, 3)
            sage: A._right_kernel_matrix()[1]
            [1 0 0]
            [0 1 0]
            [0 0 1]
       """
        tm = verbose("computing right kernel matrix over the rationals for %sx%s matrix" % (self.nrows(), self.ncols()),level=1)
        # _rational_kernel_flint() gets the zero-row case wrong, fix it there
        if self.nrows()==0:
            from .constructor import identity_matrix
            K = identity_matrix(QQ, self.ncols())
        else:
            A, _ = self._clear_denom()
            K = A._rational_kernel_iml().transpose().change_ring(QQ)
        verbose("done computing right kernel matrix over the rationals for %sx%s matrix" % (self.nrows(), self.ncols()),level=1, t=tm)
        return 'computed-iml-rational', K

    ################################################
    # Change ring
    ################################################
    def change_ring(self, R):
        """
        Create the matrix over R with entries the entries of self coerced
        into R.

        EXAMPLES::

            sage: a = matrix(QQ,2,[1/2,-1,2,3])
            sage: a.change_ring(GF(3))
            [2 2]
            [2 0]
            sage: a.change_ring(ZZ)
            Traceback (most recent call last):
            ...
            TypeError: matrix has denominators so can't change to ZZ.
            sage: b = a.change_ring(QQ['x']); b
            [1/2  -1]
            [  2   3]
            sage: b.parent()
            Full MatrixSpace of 2 by 2 dense matrices over Univariate Polynomial Ring in x over Rational Field

        TESTS:

        Make sure that subdivisions are preserved when changing rings::

            sage: a = matrix(QQ, 3, range(9))
            sage: a.subdivide(2,1); a
            [0|1 2]
            [3|4 5]
            [-+---]
            [6|7 8]
            sage: a.change_ring(ZZ).change_ring(QQ)
            [0|1 2]
            [3|4 5]
            [-+---]
            [6|7 8]
            sage: a.change_ring(GF(3))
            [0|1 2]
            [0|1 2]
            [-+---]
            [0|1 2]
        """
        if not is_Ring(R):
            raise TypeError("R must be a ring")
        from .matrix_modn_dense_double import MAX_MODULUS
        if R == self._base_ring:
            if self._is_immutable:
                return self
            return self.__copy__()
        elif is_IntegerRing(R):
            A, d = self._clear_denom()
            if d != 1:
                raise TypeError("matrix has denominators so can't change to ZZ.")
            A.subdivide(self.subdivisions())
            return A
        elif is_IntegerModRing(R) and R.order() < MAX_MODULUS:
            b = R.order()
            A, d = self._clear_denom()
            if gcd(b,d) != 1:
                raise TypeError("matrix denominator not coprime to modulus")
            B = A._mod_int(b)
            C = (1/(B.base_ring()(d))) * B
            C.subdivide(self.subdivisions())
            return C
        else:
            D = Matrix_dense.change_ring(self, R)
            D.subdivide(self.subdivisions())
            return D



    ################################################
    # Echelon form
    ################################################
    def echelonize(self, algorithm='default',
                   height_guess=None, proof=None, **kwds):
        """
        Transform the matrix ``self`` into reduced row echelon form
        in place.

        INPUT:

        -  ``algorithm``:

          - ``'default'`` (default): use heuristic choice

          - ``'padic'``: an algorithm based on the IML p-adic solver.

          - ``'multimodular'``: uses a multimodular algorithm the uses
            linbox modulo many primes.

          - ``'classical'``: just clear each column using Gauss elimination

        -  ``height_guess``, ``**kwds`` - all passed to the
           multimodular algorithm; ignored by the p-adic algorithm.

        -  ``proof`` - bool or None (default: None, see
           proof.linear_algebra or sage.structure.proof). Passed to the
           multimodular algorithm. Note that the Sage global default is
           ``proof=True``.

        OUTPUT:

        Nothing. The matrix ``self`` is transformed into reduced row
        echelon form in place.

        EXAMPLES::

            sage: a = matrix(QQ, 4, range(16)); a[0,0] = 1/19; a[0,1] = 1/5; a
            [1/19  1/5    2    3]
            [   4    5    6    7]
            [   8    9   10   11]
            [  12   13   14   15]
            sage: a.echelonize(); a
            [      1       0       0 -76/157]
            [      0       1       0  -5/157]
            [      0       0       1 238/157]
            [      0       0       0       0]

        ::

            sage: a = matrix(QQ, 4, range(16)); a[0,0] = 1/19; a[0,1] = 1/5
            sage: a.echelonize(algorithm='multimodular'); a
            [      1       0       0 -76/157]
            [      0       1       0  -5/157]
            [      0       0       1 238/157]
            [      0       0       0       0]

        TESTS:

        Echelonizing a matrix in place throws away the cache of
        the old matrix (:trac:`14506`)::

            sage: a = Matrix(QQ, [[1,2],[3,4]])
            sage: a.det(); a._clear_denom()
            -2
            (
            [1 2]
            [3 4], 1
            )
            sage: a.echelonize(algorithm="padic")
            sage: sorted(a._cache.items())
            [('in_echelon_form', True), ('pivots', (0, 1))]
            sage: a = Matrix(QQ, [[1,3],[3,4]])
            sage: a.det(); a._clear_denom()
            -5
            (
            [1 3]
            [3 4], 1
            )
            sage: a.echelonize(algorithm="multimodular")
            sage: sorted(a._cache.items())
            [('in_echelon_form', True), ('pivots', (0, 1))]
        """

        x = self.fetch('in_echelon_form')
        if not x is None: return  # already known to be in echelon form
        self.check_mutability()
        self.clear_cache()

        cdef Matrix_rational_dense E
        if algorithm == 'default':
            if self._nrows <= 6 or self._ncols <= 6:
                algorithm = "classical"
            else:
                algorithm = 'padic'

        pivots = None
        if algorithm == 'classical':
            pivots = self._echelon_in_place_classical()
        elif algorithm == 'padic':
            pivots = self._echelonize_padic()
        elif algorithm == 'multimodular':
            pivots = self._echelonize_multimodular(height_guess, proof, **kwds)
        else:
            raise ValueError("no algorithm '%s'"%algorithm)
        self.cache('in_echelon_form', True)

        if pivots is None:
            raise RuntimeError("BUG: pivots must get set")
        self.cache('pivots', tuple(pivots))


    def echelon_form(self, algorithm='default',
                     height_guess=None, proof=None, **kwds):
        r"""
        INPUT:

        -  ``algorithm``

           - 'default' (default): use heuristic choice

           - 'padic': an algorithm based on the IML p-adic solver.

           - 'multimodular': uses a multimodular algorithm the uses linbox modulo many primes.

           - 'classical': just clear each column using Gauss elimination

        -  ``height_guess``, ``**kwds`` - all passed to the
           multimodular algorithm; ignored by the p-adic algorithm.

        -  ``proof`` - bool or None (default: None, see
           proof.linear_algebra or sage.structure.proof). Passed to the
           multimodular algorithm. Note that the Sage global default is
           proof=True.


        OUTPUT: the reduced row echelon form of self.

        EXAMPLES::

            sage: a = matrix(QQ, 4, range(16)); a[0,0] = 1/19; a[0,1] = 1/5; a
            [1/19  1/5    2    3]
            [   4    5    6    7]
            [   8    9   10   11]
            [  12   13   14   15]
            sage: a.echelon_form()
            [      1       0       0 -76/157]
            [      0       1       0  -5/157]
            [      0       0       1 238/157]
            [      0       0       0       0]
            sage: a.echelon_form(algorithm='multimodular')
            [      1       0       0 -76/157]
            [      0       1       0  -5/157]
            [      0       0       1 238/157]
            [      0       0       0       0]

        The result is an immutable matrix, so if you want to
        modify the result then you need to make a copy.  This
        checks that :trac:`10543` is fixed. ::

            sage: A = matrix(QQ, 2, range(6))
            sage: E = A.echelon_form()
            sage: E.is_mutable()
            False
            sage: F = copy(E)
            sage: F[0,0] = 50
            sage: F
            [50  0 -1]
            [ 0  1  2]
        """
        label = 'echelon_form'
        x = self.fetch(label)
        if not x is None:
            return x
        if self.fetch('in_echelon_form'): return self

        if algorithm == 'default':
            if self._nrows <= 6 or self._ncols <= 6:
                algorithm = "classical"
            else:
                algorithm = 'padic'

        if algorithm == 'classical':
            E = self._echelon_classical()
        elif algorithm == 'padic':
            E = self._echelon_form_padic()
        elif algorithm == 'multimodular':
            E = self._echelon_form_multimodular(height_guess, proof=proof)
        else:
            raise ValueError("no algorithm '%s'"%algorithm)
        E.set_immutable()
        self.cache(label, E)
        self.cache('pivots', E.pivots())
        return E


    # p-adic echelonization algorithms
    def _echelon_form_padic(self, include_zero_rows=True):
        """
        Compute and return the echelon form of self using a p-adic
        nullspace algorithm.
        """
        cdef Matrix_integer_dense X
        cdef Matrix_rational_dense E
        cdef Integer d
        cdef fmpq * entry

        t = verbose('Computing echelon form of %s x %s matrix over QQ using p-adic nullspace algorithm.'%(
            self.nrows(), self.ncols()))
        A, _ = self._clear_denom()
        t = verbose('  Got integral matrix', t)
        pivots, nonpivots, X, d = A._rational_echelon_via_solve()
        t = verbose('  Computed ZZ-echelon using p-adic algorithm.', t)

        nr = self.nrows() if include_zero_rows else X.nrows()
        parent = self.matrix_space(nr, self.ncols())
        E = Matrix_rational_dense.__new__(Matrix_rational_dense, parent, None, None, None)

        # Fill in the identity part of the matrix
        cdef Py_ssize_t i, j
        for i in range(len(pivots)):
            fmpz_one(fmpq_mat_entry_num(E._matrix, i, pivots[i]))

        # Fill in the non-pivot part of the matrix
        for i in range(X.nrows()):
            for j in range(X.ncols()):
                entry = fmpq_mat_entry(E._matrix, i, nonpivots[j])
                fmpz_set(fmpq_numref(entry), fmpz_mat_entry(X._matrix, i, j))
                fmpz_set_mpz(fmpq_denref(entry), d.value)
                fmpq_canonicalise(entry)

        t = verbose('Reconstructed solution over QQ, thus completing the echelonize', t)
        E.cache('in_echelon_form', True)
        E.cache('pivots', pivots)
        return E

    def _echelonize_padic(self):
        """
        Echelonize self using a p-adic nullspace algorithm.
        """
        cdef Matrix_integer_dense X
        cdef Integer d
        cdef fmpq * entry

        t = verbose('Computing echelonization of %s x %s matrix over QQ using p-adic nullspace algorithm.'%
                    (self.nrows(), self.ncols()))
        A, _ = self._clear_denom()
        self._clear_cache()
        t = verbose('  Got integral matrix', t)
        pivots, nonpivots, X, d = A._rational_echelon_via_solve()
        t = verbose('  Computed ZZ-echelon using p-adic algorithm.', t)

        # Fill in the identity part of self.
        cdef Py_ssize_t i, j, k
        for i in range(len(pivots)):
            fmpq_one(fmpq_mat_entry(self._matrix, i, pivots[i]))

        # Fill in the non-pivot part of self.
        for i in range(X.nrows()):
            for j in range(X.ncols()):
                entry = fmpq_mat_entry(self._matrix, i, nonpivots[j])
                fmpz_set(fmpq_numref(entry), fmpz_mat_entry(X._matrix,i,j))
                fmpz_set_mpz(fmpq_denref(entry), d.value)
                fmpq_canonicalise(entry)

        # Fill in the 0-rows at the bottom.
        for i in range(len(pivots), self._nrows):
            for j in range(self._ncols):
                fmpq_zero(fmpq_mat_entry(self._matrix, i, j))

        t = verbose('Filled in echelonization of self, thus completing the echelonize', t)
        return pivots


    # Multimodular echelonization algorithms
    def _echelonize_multimodular(self, height_guess=None, proof=None, **kwds):
        cdef Matrix_rational_dense E
        E = self._echelon_form_multimodular(height_guess, proof=proof, **kwds)
        self._clear_cache()  # why!?
        fmpq_mat_set(self._matrix, E._matrix)
        return E.pivots()

    def _echelon_form_multimodular(self, height_guess=None, proof=None):
        """
        Return reduced row-echelon form using a multi-modular algorithm.
        This does not change ``self``.

        REFERENCE:

        - Chapter 7 of Stein's "Explicitly Computing Modular Forms".

        INPUT:


        -  ``height_guess`` - integer or None

        -  ``proof`` - boolean (default: None, see
           proof.linear_algebra or sage.structure.proof) Note that the Sage
           global default is proof=True.
        """
        from .misc import matrix_rational_echelon_form_multimodular
        return matrix_rational_echelon_form_multimodular(self,
                                 height_guess=height_guess, proof=proof)

    cdef swap_rows_c(self, Py_ssize_t r1, Py_ssize_t r2):
        """
        EXAMPLES::

            sage: a = matrix(QQ,2,[1..6])
            sage: a.swap_rows(0,1)             # indirect doctest
            sage: a
            [4 5 6]
            [1 2 3]
        """
        # no bounds checking!
        cdef Py_ssize_t c
        for c in range(self._ncols):
            fmpq_swap(fmpq_mat_entry(self._matrix, r1, c),
                      fmpq_mat_entry(self._matrix, r2, c))

    cdef swap_columns_c(self, Py_ssize_t c1, Py_ssize_t c2):
        """
        EXAMPLES::

            sage: a = matrix(QQ,2,[1..6])
            sage: a.swap_columns(0,1)          # indirect doctest
            sage: a
            [2 1 3]
            [5 4 6]
        """
        # no bounds checking!
        for r in range(self._nrows):
            fmpq_swap(fmpq_mat_entry(self._matrix, r, c1),
                      fmpq_mat_entry(self._matrix, r, c2))

    def decomposition(self, is_diagonalizable=False, dual=False,
                      algorithm='default', height_guess=None, proof=None):
        """
        Returns the decomposition of the free module on which this matrix A
        acts from the right (i.e., the action is x goes to x A), along with
        whether this matrix acts irreducibly on each factor. The factors
        are guaranteed to be sorted in the same way as the corresponding
        factors of the characteristic polynomial.

        Let A be the matrix acting from the on the vector space V of column
        vectors. Assume that A is square. This function computes maximal
        subspaces W_1, ..., W_n corresponding to Galois conjugacy classes
        of eigenvalues of A. More precisely, let f(X) be the characteristic
        polynomial of A. This function computes the subspace
        `W_i = ker(g_(A)^n)`, where g_i(X) is an irreducible
        factor of f(X) and g_i(X) exactly divides f(X). If the optional
        parameter is_diagonalizable is True, then we let W_i = ker(g(A)),
        since then we know that ker(g(A)) = `ker(g(A)^n)`.

        If dual is True, also returns the corresponding decomposition of V
        under the action of the transpose of A. The factors are guaranteed
        to correspond.

        INPUT:


        -  ``is_diagonalizable`` - ignored

        -  ``dual`` - whether to also return decompositions for
           the dual

        -  ``algorithm``

           - 'default': use default algorithm for computing Echelon
             forms

           - 'multimodular': much better if the answers
             factors have small height

        -  ``height_guess`` - positive integer; only used by
           the multimodular algorithm

        -  ``proof`` - bool or None (default: None, see
           proof.linear_algebra or sage.structure.proof); only used by the
           multimodular algorithm. Note that the Sage global default is
           proof=True.


        .. NOTE::

           IMPORTANT: If you expect that the subspaces in the answer
           are spanned by vectors with small height coordinates, use
           algorithm='multimodular' and height_guess=1; this is
           potentially much faster than the default. If you know for a
           fact the answer will be very small, use
           algorithm='multimodular', height_guess=bound on height,
           proof=False.

        You can get very very fast decomposition with proof=False.

        EXAMPLES::

            sage: a = matrix(QQ,3,[1..9])
            sage: a.decomposition()
            [
            (Vector space of degree 3 and dimension 1 over Rational Field
            Basis matrix:
            [ 1 -2  1], True),
            (Vector space of degree 3 and dimension 2 over Rational Field
            Basis matrix:
            [ 1  0 -1]
            [ 0  1  2], True)
            ]

        """
        X = self._decomposition_rational(is_diagonalizable=is_diagonalizable,
                                         echelon_algorithm = algorithm,
                                         height_guess = height_guess, proof=proof)
        if dual:
            Y = self.transpose()._decomposition_rational(is_diagonalizable=is_diagonalizable,
                   echelon_algorithm = algorithm, height_guess = height_guess, proof=proof)
            return X, Y
        return X

    def _decomposition_rational(self, is_diagonalizable = False,
                                echelon_algorithm='default',
                                kernel_algorithm='default',
                                **kwds):
        """
        Returns the decomposition of the free module on which this matrix A
        acts from the right (i.e., the action is x goes to x A), along with
        whether this matrix acts irreducibly on each factor. The factors
        are guaranteed to be sorted in the same way as the corresponding
        factors of the characteristic polynomial.

        INPUT:


        -  ``self`` - a square matrix over the rational
           numbers

        -  ``echelon_algorithm`` - 'default'

        -  ``'multimodular'`` - use this if the answers have
           small height

        -  ``**kwds`` - passed on to echelon function.

        .. NOTE::

           IMPORTANT: If you expect that the subspaces in the answer are
           spanned by vectors with small height coordinates, use
           algorithm='multimodular' and height_guess=1; this is potentially
           much faster than the default. If you know for a fact the answer
           will be very small, use algorithm='multimodular',
           height_guess=bound on height, proof=False


        OUTPUT:


        -  ``Sequence`` - list of tuples (V,t), where V is a
           vector spaces and t is True if and only if the charpoly of self on
           V is irreducible. The tuples are in order corresponding to the
           elements of the sorted list self.charpoly().factor().
        """
        cdef Py_ssize_t k

        if not self.is_square():
            raise ArithmeticError("self must be a square matrix")

        if self.nrows() == 0:
            return decomp_seq([])

        A, _ = self._clear_denom()

        f = A.charpoly('x')
        E = decomp_seq([])

        t = verbose('factoring the characteristic polynomial', level=2, caller_name='rational decomp')
        F = f.factor()
        verbose('done factoring', t=t, level=2, caller_name='rational decomp')

        if len(F) == 1:
            V = QQ**self.nrows()
            m = F[0][1]
            return decomp_seq([(V, m==1)])

        V = ZZ**self.nrows()
        v = V.random_element()
        num_iterates = max([0] + [f.degree() - g.degree() for g, _ in F if g.degree() > 1]) + 1

        S = [ ]

        F.sort()
        for i in range(len(F)):
            g, m = F[i]

            if g.degree() == 1:
                # Just use kernel -- much easier.
                B = A.__copy__()
                for k from 0 <= k < A.nrows():
                    B[k,k] += g[0]
                if m > 1 and not is_diagonalizable:
                    B = B**m
                B = B.change_ring(QQ)
                W = B.kernel(algorithm = kernel_algorithm, **kwds)
                E.append((W, m==1))
                continue

            # General case, i.e., deg(g) > 1:
            W = None
            tries = m
            while True:

                # Compute the complementary factor of the charpoly.
                h = f // (g**m)
                v = h.list()

                while len(S) < tries:
                    t = verbose('%s-spinning %s-th random vector'%(num_iterates, len(S)),
                                level=2, caller_name='rational decomp')
                    S.append(A.iterates(V.random_element(x=-10,y=10), num_iterates))
                    verbose('done spinning', level=2, t=t, caller_name='rational decomp')

                for j in range(0 if W is None else W.nrows() // g.degree(), len(S)):
                    # Compute one element of the kernel of g(A)**m.
                    t = verbose('compute element of kernel of g(A), for g of degree %s'%g.degree(),level=2,
                            caller_name='rational decomp')
                    w = S[j].linear_combination_of_rows(h.list())
                    t = verbose('done computing element of kernel of g(A)', t=t,level=2, caller_name='rational decomp')

                    # Get the rest of the kernel.
                    t = verbose('fill out rest of kernel',level=2, caller_name='rational decomp')
                    if W is None:
                        W = A.iterates(w, g.degree())
                    else:
                        W = W.stack(A.iterates(w, g.degree()))
                    t = verbose('finished filling out more of kernel',level=2, t=t, caller_name='rational decomp')

                if W.rank() == m * g.degree():
                    W = W.change_ring(QQ)
                    t = verbose('now computing row space', level=2, caller_name='rational decomp')
                    W.echelonize(algorithm = echelon_algorithm, **kwds)
                    E.append((W.row_space(), m==1))
                    verbose('computed row space', level=2,t=t, caller_name='rational decomp')
                    break
                else:
                    verbose('we have not yet generated all the kernel (rank so far=%s, target rank=%s)'%(
                        W.rank(), m*g.degree()), level=2, caller_name='rational decomp')
                    tries += 1
                    if tries > 5*m:
                        raise RuntimeError("likely bug in decomposition")
                # end if
            #end while
        #end for
        return decomp_seq(E)


##     def simple_decomposition(self, echelon_algorithm='default', **kwds):
##         """
##         Returns the decomposition of the free module on which this
##         matrix A acts from the right (i.e., the action is x goes to x
##         A), as a direct sum of simple modules.

##         NOTE: self *must* be diagonalizable.

##         INPUT:
##             self -- a square matrix that is assumed to be diagonalizable
##             echelon_algorithm -- 'default'
##                                  'multimodular' -- use this if the answers
##                                  have small height
##             **kwds -- passed on to echelon function.

##         IMPORTANT NOTE:
##         If you expect that the subspaces in the answer are spanned by vectors
##         with small height coordinates, use algorithm='multimodular' and
##         height_guess=1; this is potentially much faster than the default.
##         If you know for a fact the answer will be very small, use
##            algorithm='multimodular', height_guess=bound on height, proof=False

##         OUTPUT:
##             Sequence -- list of tuples (V,g), where V is a subspace
##                         and an irreducible polynomial g, which is the
##                         charpoly (=minpoly) of self acting on V.
##         """
##         cdef Py_ssize_t k

##         if not self.is_square():
##             raise ArithmeticError("self must be a square matrix")

##         if self.nrows() == 0:
##             return decomp_seq([])

##         A, _ = self._clear_denom()

##         f = A.charpoly('x')
##         E = decomp_seq([])

##         t = verbose('factoring the characteristic polynomial', level=2, caller_name='simple decomp')
##         F = f.factor()
##         G = [g for g, _ in F]
##         minpoly = prod(G)
##         squarefree_degree = sum([g.degree() for g in G])
##         verbose('done factoring', t=t, level=2, caller_name='simple decomp')

##         V = ZZ**self.nrows()
##         v = V.random_element()
##         num_iterates = max([squarefree_degree - g.degree() for g in G]) + 1

##         S = [ ]

##         F.sort()
##         for i in range(len(F)):
##             g, m = F[i]

##             if g.degree() == 1:
##                 # Just use kernel -- much easier.
##                 B = A.__copy__()
##                 for k from 0 <= k < A.nrows():
##                     B[k,k] += g[0]
##                 if m > 1 and not is_diagonalizable:
##                     B = B**m
##                 W = B.change_ring(QQ).kernel()
##                 for b in W.basis():
##                     E.append((W.span(b), g))
##                 continue

##             # General case, i.e., deg(g) > 1:
##             W = None
##             while True:

##                 # Compute the complementary factor of the charpoly.
##                 h = minpoly // g
##                 v = h.list()

##                 while len(S) < m:
##                     t = verbose('%s-spinning %s-th random vector'%(num_iterates, len(S)),
##                                 level=2, caller_name='simple decomp')
##                     S.append(A.iterates(V.random_element(x=-10,y=10), num_iterates))
##                     verbose('done spinning', level=2, t=t, caller_name='simple decomp')

##                 for j in range(len(S)):
##                     # Compute one element of the kernel of g(A).
##                     t = verbose('compute element of kernel of g(A), for g of degree %s'%g.degree(),level=2,
##                             caller_name='simple decomp')
##                     w = S[j].linear_combination_of_rows(h.list())
##                     t = verbose('done computing element of kernel of g(A)', t=t,level=2, caller_name='simple decomp')

##                     # Get the rest of the kernel.
##                     t = verbose('fill out rest of kernel',level=2, caller_name='simple decomp')
##                     if W is None:
##                         W = A.iterates(w, g.degree())
##                     else:
##                         W = W.stack(A.iterates(w, g.degree()))
##                     t = verbose('finished filling out more of kernel',level=2, t=t, caller_name='simple decomp')

##                 if W.rank() == m * g.degree():
##                     W = W.change_ring(QQ)
##                     t = verbose('now computing row space', level=2, caller_name='simple decomp')
##                     W.echelonize(algorithm = echelon_algorithm, **kwds)
##                     E.append((W.row_space(), m==1))
##                     verbose('computed row space', level=2,t=t, caller_name='simple decomp')
##                     break
##                 else:
##                     verbose('we have not yet generated all the kernel (rank so far=%s, target rank=%s)'%(
##                         W.rank(), m*g.degree()), level=2, caller_name='simple decomp')
##                     j += 1
##                     if j > 3*m:
##                         raise RuntimeError("likely bug in decomposition")
##                 # end if
##             #end while
##         #end for
##         return E


    def _lift_crt_rr(self, res, mm):
        cdef Integer m
        cdef Matrix_integer_dense ZA
        cdef Matrix_rational_dense QA
        cdef Py_ssize_t i, j
        cdef mpz_t* Z_row
        cdef mpq_t* Q_row
        cdef mpz_t tmp
        cdef mpq_t tmp2
        mpz_init(tmp)
        mpq_init(tmp2)
        ZA = _lift_crt(res, mm)
        QA = Matrix_rational_dense.__new__(Matrix_rational_dense, self.parent(), None, None, None)
        m = mm.prod()
        for i in range(ZA._nrows):
            for j in range(ZA._ncols):
                fmpz_get_mpz(tmp, fmpz_mat_entry(ZA._matrix,i,j))
                mpq_rational_reconstruction(tmp2, tmp, m.value)
                fmpq_set_mpq(fmpq_mat_entry(QA._matrix, i, j), tmp2)
        mpz_clear(tmp)
        mpq_clear(tmp2)
        return QA

    def randomize(self, density=1, num_bound=2, den_bound=2, \
                  distribution=None, nonzero=False):
        """
        Randomize ``density`` proportion of the entries of this matrix, leaving
        the rest unchanged.

        If ``x`` and ``y`` are given, randomized entries of this matrix have
        numerators and denominators bounded by ``x`` and ``y`` and have
        density 1.

        INPUT:

        -  ``density`` - number between 0 and 1 (default: 1)

        -  ``num_bound`` - numerator bound (default: 2)

        -  ``den_bound`` - denominator bound (default: 2)

        -  ``distribution`` - ``None`` or '1/n' (default: ``None``); if '1/n'
           then ``num_bound``, ``den_bound`` are ignored and numbers are chosen
           using the GMP function ``mpq_randomize_entry_recip_uniform``

        OUTPUT:

        -  None, the matrix is modified in-space

        EXAMPLES::

            sage: a = matrix(QQ,2,4); a.randomize(); a
            [ 0 -1  2 -2]
            [ 1 -1  2  1]
            sage: a = matrix(QQ,2,4); a.randomize(density=0.5); a
            [ -1  -2   0   0]
            [  0   0 1/2   0]
            sage: a = matrix(QQ,2,4); a.randomize(num_bound=100, den_bound=100); a
            [ 14/27  21/25  43/42 -48/67]
            [-19/55  64/67 -11/51     76]
            sage: a = matrix(QQ,2,4); a.randomize(distribution='1/n'); a
            [      3     1/9     1/2     1/4]
            [      1    1/39       2 -1955/2]

        TESTS:

        Check that the option ``nonzero`` is meaningful (:trac:`22970`)::

            sage: a = matrix(QQ, 10, 10, 1)
            sage: b = a.__copy__()
            sage: b.randomize(nonzero=True)
            sage: a == b
            False
            sage: any(b[i,j].is_zero() for i in range(10) for j in range(10))
            False
        """
        density = float(density)
        if density <= 0.0:
            return

        self.check_mutability()
        self.clear_cache()

        cdef Integer B, C
        cdef Py_ssize_t i, j, k, num_per_row
        cdef randstate rstate
        cdef mpq_t tmp

        B = Integer(num_bound + 1)
        C = Integer(den_bound + 1)

        mpq_init(tmp)

        if not nonzero:
            if density >= 1.0:
                if distribution == "1/n":
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry_recip_uniform(tmp)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
                elif mpz_cmp_si(C.value, 2):   # denom is > 1
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry(tmp, B.value, C.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
                else:
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry_as_int(tmp, B.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
            else:
                rstate = current_randstate()
                num_per_row = int(density * self._ncols)
                if distribution == "1/n":
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry_recip_uniform(tmp)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()
                elif mpz_cmp_si(C.value, 2):   # denom is > 1
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry(tmp, B.value, C.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()
                else:
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry_as_int(tmp, B.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()
        else:
            if density >= 1.0:
                if distribution == "1/n":
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry_recip_uniform_nonzero(tmp)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
                elif mpz_cmp_si(C.value, 2):   # denom is > 1
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry_nonzero(tmp, B.value, C.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
                else:
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(self._ncols):
                            mpq_randomize_entry_as_int_nonzero(tmp, B.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, j), tmp)
                    sig_off()
            else:
                rstate = current_randstate()
                num_per_row = int(density * self._ncols)
                if distribution == "1/n":
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry_recip_uniform_nonzero(tmp)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()
                elif mpz_cmp_si(C.value, 2):   # denom is > 1
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry_nonzero(tmp, B.value, C.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()
                else:
                    sig_on()
                    for i in range(self._nrows):
                        for j in range(num_per_row):
                            k = rstate.c_random() % self._ncols
                            mpq_randomize_entry_as_int_nonzero(tmp, B.value)
                            fmpq_set_mpq(fmpq_mat_entry(self._matrix, i, k), tmp)
                    sig_off()

        mpq_clear(tmp)


    def rank(self):
        """
        Return the rank of this matrix.

        EXAMPLES::
            sage: matrix(QQ,3,[1..9]).rank()
            2
            sage: matrix(QQ,100,[1..100^2]).rank()
            2
        """
        r = self.fetch('rank')
        if not r is None: return r
        if self._nrows <= 6 and self._ncols <= 6 and self.height().ndigits() <= 10000:
            r = self._rank_pari()
        else:
            A, _ = self._clear_denom()
            r = A.rank()
        self.cache('rank', r)
        return r

    def transpose(self):
        """
        Returns the transpose of self, without changing self.

        EXAMPLES:

        We create a matrix, compute its transpose, and note that the
        original matrix is not changed.

        ::

            sage: A = matrix(QQ, 2, 3, range(6))
            sage: type(A)
            <type 'sage.matrix.matrix_rational_dense.Matrix_rational_dense'>
            sage: B = A.transpose()
            sage: print(B)
            [0 3]
            [1 4]
            [2 5]
            sage: print(A)
            [0 1 2]
            [3 4 5]

        ``.T`` is a convenient shortcut for the transpose::

            sage: print(A.T)
            [0 3]
            [1 4]
            [2 5]

        ::

            sage: A.subdivide(None, 1); A
            [0|1 2]
            [3|4 5]
            sage: A.transpose()
            [0 3]
            [---]
            [1 4]
            [2 5]
        """
        cdef Matrix_rational_dense ans
        if self._nrows == self._ncols:
            parent = self._parent
        else:
            parent = self._parent.matrix_space(self._ncols, self._nrows)
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, parent, None, None, None)
        sig_on()
        fmpq_mat_transpose(ans._matrix, self._matrix)
        sig_off()

        if self._subdivisions is not None:
            row_divs, col_divs = self.subdivisions()
            ans.subdivide(col_divs, row_divs)
        return ans

    def antitranspose(self):
        """
        Returns the antitranspose of self, without changing self.

        EXAMPLES::

            sage: A = matrix(QQ,2,3,range(6))
            sage: type(A)
            <type 'sage.matrix.matrix_rational_dense.Matrix_rational_dense'>
            sage: A.antitranspose()
            [5 2]
            [4 1]
            [3 0]
            sage: A
            [0 1 2]
            [3 4 5]

            sage: A.subdivide(1,2); A
            [0 1|2]
            [---+-]
            [3 4|5]
            sage: A.antitranspose()
            [5|2]
            [-+-]
            [4|1]
            [3|0]
        """
        if self._nrows == self._ncols:
            parent = self._parent
        else:
            parent = self._parent.matrix_space(self._ncols, self._nrows)

        cdef Matrix_rational_dense ans
        ans = Matrix_rational_dense.__new__(Matrix_rational_dense, parent, None, None, None)

        cdef Py_ssize_t i,j
        cdef Py_ssize_t ri,rj # reversed i and j
        sig_on()
        ri = self._nrows
        for i in range(self._nrows):
            rj = self._ncols
            ri =  ri - 1
            for j in range(self._ncols):
                rj = rj - 1
                fmpq_set(fmpq_mat_entry(ans._matrix, rj, ri),
                         fmpq_mat_entry(self._matrix, i, j))
        sig_off()

        if self._subdivisions is not None:
            row_divs, col_divs = self.subdivisions()
            ans.subdivide([self._ncols - t for t in reversed(col_divs)],
                        [self._nrows - t for t in reversed(row_divs)])
        return ans

    def set_row_to_multiple_of_row(self, Py_ssize_t i, Py_ssize_t j, s):
        """
        Set row i equal to s times row j.

        EXAMPLES::

            sage: a = matrix(QQ,2,3,range(6)); a
            [0 1 2]
            [3 4 5]
            sage: a.set_row_to_multiple_of_row(1,0,-3)
            sage: a
            [ 0  1  2]
            [ 0 -3 -6]
        """
        self.check_row_bounds_and_mutability(i, j)
        cdef Py_ssize_t k
        cdef fmpq_t ss
        fmpq_init(ss)
        fmpq_set_mpq(ss, (<Rational> Rational(s)).value)
        for k in range(self._ncols):
            fmpq_mul(fmpq_mat_entry(self._matrix, i, k),
                     fmpq_mat_entry(self._matrix, j, k),
                     ss)
        fmpq_clear(ss)

    def _set_row_to_negative_of_row_of_A_using_subset_of_columns(self, Py_ssize_t i, Matrix A,
                                                                 Py_ssize_t r, cols,
                                                                 cols_index=None):
        """
        Set row i of self to -(row r of A), but where we only take the
        given column positions in that row of A. We do not zero out the
        other entries of self's row i either.


        .. NOTE::

            This function exists just because it is useful for modular symbols presentations.

        INPUT:


        -  ``i`` - integer, index into the rows of self

        -  ``A`` - a matrix

        -  ``r`` - integer, index into rows of A

        -  ``cols`` - a *sorted* list of integers.


        EXAMPLES::

            sage: a = matrix(QQ,2,3,range(6)); a
            [0 1 2]
            [3 4 5]
            sage: a._set_row_to_negative_of_row_of_A_using_subset_of_columns(0,a,1,[1,2])
            sage: a
            [-4 -5  2]
            [ 3  4  5]
        """
        self.check_row_bounds_and_mutability(i,i)
        cdef Matrix_rational_dense _A
        if r < 0 or r >= A.nrows():
            raise IndexError("invalid row")
        cdef Py_ssize_t l = 0

        if not A.base_ring() == QQ:
            A = A.change_ring(QQ)
        if not A.is_dense():
            A = A.dense_matrix()

        _A = A
        for k in cols:
            entry = fmpq_mat_entry(self._matrix, i, l)
            fmpq_set(entry, fmpq_mat_entry(_A._matrix, r, k))
            fmpq_neg(entry, entry)
            l += 1


    def _add_col_j_of_A_to_col_i_of_self(self,
               Py_ssize_t i, Matrix_rational_dense A, Py_ssize_t j):
        """
        Unsafe technical function that very quickly adds the j-th column of
        A to the i-th column of self.

        Does not check mutability.
        """
        if A._nrows != self._nrows:
            raise TypeError("nrows of self and A must be the same")
        cdef Py_ssize_t r
        for r in range(self._nrows):
            fmpq_add(fmpq_mat_entry(self._matrix, r, i),
                     fmpq_mat_entry(self._matrix, r, i),
                     fmpq_mat_entry(A._matrix, r, j))


    def _det_pari(self, int flag=0):
        """
        Return the determinant of this matrix computed using pari.

        EXAMPLES::
            sage: matrix(QQ,3,[1..9])._det_pari()
            0
            sage: matrix(QQ,3,[1..9])._det_pari(1)
            0
            sage: matrix(QQ,3,[0]+[2..9])._det_pari()
            3
        """
        if self._nrows != self._ncols:
            raise ValueError("self must be a square matrix")
        sig_on()
        cdef GEN d = det0(_new_GEN_from_fmpq_mat_t(self._matrix), flag)
        # now convert d to a Sage rational
        cdef Rational e = <Rational> Rational.__new__(Rational)
        INTFRAC_to_mpq(e.value, d)
        clear_stack()
        return e

    def _rank_pari(self):
        """
        Return the rank of this matrix computed using pari.

        EXAMPLES::

            sage: matrix(QQ,3,[1..9])._rank_pari()
            2
        """
        sig_on()
        cdef long r = rank(_new_GEN_from_fmpq_mat_t(self._matrix))
        clear_stack()
        return r

    def _multiply_pari(self, Matrix_rational_dense right):
        """
        Return the product of self and right, computed using PARI.

        EXAMPLES::

            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9])._multiply_pari(matrix(QQ,2,[1,2,3,4]))
            [  -9/5 -34/15]
            [ 25/12  59/18]

        We verify that 0 rows or columns works::

            sage: x = matrix(QQ,2,0); y= matrix(QQ,0,2); x*y
            [0 0]
            [0 0]
            sage: matrix(ZZ, 0, 0)*matrix(QQ, 0, 5)
            []
        """
        if self._ncols != right._nrows:
            raise ArithmeticError("self must be a square matrix")
        if not self._ncols*self._nrows or not right._ncols*right._nrows:
            # pari doesn't work in case of 0 rows or columns
            # This case is easy, since the answer must be the 0 matrix.
            return self.matrix_space(self._nrows, right._ncols).zero_matrix().__copy__()
        sig_on()
        cdef GEN M = gmul(_new_GEN_from_fmpq_mat_t(self._matrix),
                          _new_GEN_from_fmpq_mat_t(right._matrix))
        A = new_matrix_from_pari_GEN(self.matrix_space(self._nrows, right._ncols), M)
        clear_stack()
        return A

    def _invert_pari(self):
        """
        Return the inverse of this matrix computed using PARI.

        EXAMPLES::

            sage: matrix(QQ,2,[1,2,3,4])._invert_pari()
            [  -2    1]
            [ 3/2 -1/2]
            sage: matrix(QQ,2,[1,2,2,4])._invert_pari()
            Traceback (most recent call last):
            ...
            PariError: impossible inverse in ginv: [1, 2; 2, 4]
        """
        if self._nrows != self._ncols:
            raise ValueError("self must be a square matrix")
        cdef GEN M, d

        sig_on()
        M = _new_GEN_from_fmpq_mat_t(self._matrix)
        d = ginv(M)

        # Convert matrix back to Sage.
        A = new_matrix_from_pari_GEN(self._parent, d)
        clear_stack()
        return A

    def __pari__(self):
        """
        Return pari version of this matrix.

        EXAMPLES::

            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).__pari__()
            [1/5, -2/3; 3/4, 4/9]
        """
        return rational_matrix(self._matrix, False)

    def row(self, Py_ssize_t i, from_list=False):
        """
        Return the i-th row of this matrix as a dense vector.

        INPUT:

            -  ``i`` - integer
            -  ``from_list`` - ignored

        EXAMPLES::

            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).row(1)
            (3/4, 4/9)
            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).row(1,from_list=True)
            (3/4, 4/9)
            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).row(-2)
            (1/5, -2/3)
        """
        if self._nrows == 0:
            raise IndexError("matrix has no rows")
        if i >= self._nrows or i < -self._nrows:
            raise IndexError("row index out of range")
        if i < 0:
            i = i + self._nrows
        cdef Py_ssize_t j
        from sage.modules.free_module import FreeModule
        parent = FreeModule(self._base_ring, self._ncols)
        cdef Vector_rational_dense v = Vector_rational_dense.__new__(Vector_rational_dense)
        v._init(self._ncols, parent)
        for j in range(self._ncols):
            fmpq_get_mpq(v._entries[j], fmpq_mat_entry(self._matrix, i, j))
        return v

    def column(self, Py_ssize_t i, from_list=False):
        """
        Return the i-th column of this matrix as a dense vector.

        INPUT:
            -  ``i`` - integer
            -  ``from_list`` - ignored

        EXAMPLES::

            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).column(1)
            (-2/3, 4/9)
            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).column(1,from_list=True)
            (-2/3, 4/9)
            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).column(-1)
            (-2/3, 4/9)
            sage: matrix(QQ,2,[1/5,-2/3,3/4,4/9]).column(-2)
            (1/5, 3/4)
        """
        if self._ncols == 0:
            raise IndexError("matrix has no columns")
        if i >= self._ncols or i < -self._ncols:
            raise IndexError("row index out of range")
        if i < 0:
            i += self._ncols
        cdef Py_ssize_t j
        from sage.modules.free_module import FreeModule
        parent = FreeModule(self._base_ring, self._nrows)
        cdef Vector_rational_dense v = Vector_rational_dense.__new__(Vector_rational_dense)
        v._init(self._nrows, parent)
        for j in range(self._nrows):
            fmpq_get_mpq(v._entries[j], fmpq_mat_entry(self._matrix, j, i))
        return v

    ################################################
    # LLL
    ################################################

    def LLL(self, *args, **kwargs):
        """
        Return an LLL reduced or approximated LLL reduced lattice for
        ``self`` interpreted as a lattice.

        For details on input parameters, see
        :meth:`sage.matrix.matrix_integer_dense.Matrix_integer_dense.LLL`.

        EXAMPLES::

            sage: A = Matrix(QQ, 3, 3, [1/n for n in range(1, 10)])
            sage: A.LLL()
            [ 1/28 -1/40 -1/18]
            [ 1/28 -1/40  1/18]
            [    0 -3/40     0]
        """
        A, d = self._clear_denom()
        return A.LLL(*args, **kwargs) / d


cdef new_matrix_from_pari_GEN(parent, GEN d):
    """
    Given a PARI GEN with ``t_INT`` or ``t_FRAC entries, create a
    :class:`Matrix_rational_dense` from it.

    EXAMPLES::

        sage: matrix(QQ,2,[1..4])._multiply_pari(matrix(QQ,2,[2..5]))       # indirect doctest
        [10 13]
        [22 29]
    """
    cdef Py_ssize_t i, j
    cdef Matrix_rational_dense B = Matrix_rational_dense.__new__(
        Matrix_rational_dense, parent, None, None, None)
    cdef mpq_t tmp
    mpq_init(tmp)
    for i in range(B._nrows):
        for j in range(B._ncols):
            INTFRAC_to_mpq(tmp, gcoeff(d, i+1, j+1))
            fmpq_set_mpq(fmpq_mat_entry(B._matrix, i, j), tmp)
    mpq_clear(tmp)
    return B
