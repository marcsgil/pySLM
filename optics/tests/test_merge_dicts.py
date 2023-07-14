import unittest
import numpy.testing as npt

from optics.utils import merge_dicts

import numpy as np


def dicteq(a, b):
    """
    Compares two dictionaries recursively
    :param a:
    :param b:
    :return: boolean, True if identical
    """
    identical = True
    for k in set(a.keys()).union(set(b.keys())):
        if k in a.keys():
            a_k = a[k]
            if k in b.keys():
                b_k = b[k]
                if isinstance(a_k, dict):
                    if isinstance(b_k, dict):
                        identical = dicteq(a_k, b_k)
                    else:
                        identical = False
                elif isinstance(b_k, dict):
                    identical = False
                else:
                    if type(a_k) == type(b_k):
                        if len(a_k) == len(b_k):
                            identical = np.all(a_k == b_k)
                        else:
                            identical = False
                    else:
                        identical = False
            else:
                identical = False
        else:
            identical = False
        if not identical:
            break

    return identical


class TestMergeDicts(unittest.TestCase):
    def setUp(self):
        self.array0 = np.zeros(4)
        self.array1 = np.ones(5)
        self.defaults = dict(a='a', b='b', c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array0)
        self.missing = dict(a='a', c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array0)
        self.additional = dict(a='a', b='b', B='B', c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array0)
        self.replace0 = dict(a='A', b='b', c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array0)
        self.replace1 = dict(a='a', b='b', c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array1)
        self.branch = dict(a=dict(A='A', B=dict(c=[0, 1, 2], D='D')), c=[0, 1, 2], d=dict(e='e', f='f', g=dict(h='h', i='i')), j=dict(k='k'), l=self.array0)

    def test_eq(self):
        npt.assert_equal(dicteq(self.defaults, self.defaults), True,
                         err_msg='Testing of comparison function failed for defaults.')
        npt.assert_equal(dicteq(self.defaults, self.defaults), True,
                         err_msg='Testing of comparison function failed for defaults in reverse.')
        npt.assert_equal(dicteq(self.missing, self.defaults), False,
                         err_msg='Testing of comparison function failed for missing.')
        npt.assert_equal(dicteq(self.defaults, self.missing), False,
                         err_msg='Testing of comparison function failed for missing in reverse.')
        npt.assert_equal(dicteq(self.additional, self.defaults), False,
                         err_msg='Testing of comparison function failed for additional.')
        npt.assert_equal(dicteq(self.defaults, self.additional), False,
                         err_msg='Testing of comparison function failed for additional in reverse.')
        npt.assert_equal(dicteq(self.replace0, self.defaults), False,
                         err_msg='Testing of comparison function failed for replacement.')
        npt.assert_equal(dicteq(self.defaults, self.replace0), False,
                         err_msg='Testing of comparison function failed for replacement in reverse.')
        npt.assert_equal(dicteq(self.replace1, self.defaults), False,
                         err_msg='Testing of comparison function failed for replacement.')
        npt.assert_equal(dicteq(self.defaults, self.replace1), False,
                         err_msg='Testing of comparison function failed for replacement in reverse.')
        npt.assert_equal(dicteq(self.branch, self.defaults), False,
                         err_msg='Testing of comparison function failed for branch addition.')
        npt.assert_equal(dicteq(self.defaults, self.branch), False,
                         err_msg='Testing of comparison function failed for branch addition in reverse.')

    def test_self(self):
        result = merge_dicts(self.defaults, self.defaults)
        npt.assert_equal(dicteq(result, self.defaults), True, err_msg='merging default with itself failed')

    def test_missing(self):
        result = merge_dicts(self.missing, self.defaults)
        npt.assert_equal(dicteq(result, self.defaults), True, err_msg="merging didn't replace missing")

    def test_additional(self):
        result = merge_dicts(self.additional, self.defaults)
        npt.assert_equal(dicteq(result, self.additional), True, err_msg="merging didn't add new")

    def test_replace(self):
        result = merge_dicts(self.replace0, self.defaults)
        npt.assert_equal(dicteq(result, self.replace0), True, err_msg="merging didn't keep replaced string")
        result = merge_dicts(self.replace1, self.defaults)
        npt.assert_equal(dicteq(result, self.replace1), True, err_msg="merging didn't keep replaced ndarray")


if __name__ == '__main__':
    unittest.main()
