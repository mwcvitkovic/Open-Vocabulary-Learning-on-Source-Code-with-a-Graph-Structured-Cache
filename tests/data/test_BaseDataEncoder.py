# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import re
import unittest

from hypothesis import given, strategies, example

from data.BaseDataEncoder import BaseDataEncoder

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
decoder = [str(i) for i in range(10)] + ['_'] + [chr(i) for i in range(97, 123)]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestBaseDataEncoder(unittest.TestCase):
    @given(s=strategies.text())
    @example(s='')
    @example(s='_')
    @example(s='__')
    def test_name_to_subtokens(self, s):
        st = BaseDataEncoder.name_to_subtokens(s)
        self.assertTrue(''.join(st) == s.lower().replace('_', ''))

    def test_name_to_subtokens_2(self):
        s = 'AAA'
        st = BaseDataEncoder.name_to_subtokens(s)
        self.assertTrue(st == ['aaa'])

    @given(s=strategies.text())
    @example(s='0_9Az$')
    @example(s='')
    @example(s='_')
    @example(s='__')
    @example(s='AAA')
    def test_name_to_1_hot(self, s):
        for size in [5, 1, 31, 100]:
            for special in [True, False]:
                for internal in [True, False]:
                    one_hot = BaseDataEncoder.name_to_1_hot(s, size, special, internal)
                    decoded = BaseDataEncoder.name_from_1_hot(one_hot)
                    if special:
                        self.assertTrue(all(one_hot[38, :] == 1))
                        self.assertTrue(one_hot.sum() == one_hot.shape[1])
                    else:
                        self.assertTrue(all(one_hot[38, :] == 0))
                    if internal and not special:
                        self.assertTrue(all(one_hot[39, :] == 1))
                        self.assertTrue(one_hot.sum() == one_hot.shape[1])
                    else:
                        self.assertTrue(all(one_hot[39, :] == 0))
                    if not special and not internal:
                        self.assertNotIn('S', decoded)
                        self.assertNotIn('I', decoded)
                        orig = first_cap_re.sub(r'\1_\2', s)
                        orig = all_cap_re.sub(r'\1_\2', orig).lower()
                        orig = orig[:size]
                        self.assertTrue(all(
                            decoded[i] == orig[i] if decoded[i] != 'U' else decoded[i] not in decoder for i in
                            range(len(decoded))))
