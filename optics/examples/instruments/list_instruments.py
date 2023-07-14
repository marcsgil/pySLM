#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Demonstration of the list() method on Instrument.
# This recursively prints a tree of all the instruments that can be controlled with this library.
#

from examples.instruments import log
import optics.instruments

if __name__ == '__main__':
    all_instruments = optics.instruments.Instrument.list(recursive=True)

    log.info('All instruments:')
    for instr_descriptor in all_instruments:
        log.info(instr_descriptor)
