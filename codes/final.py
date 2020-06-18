'''
'''

import numpy as np
import pandas as pd
import sqlite3
import matplotlib as mpl
import matplotlib.pyplot as plt

class something:
    '''
    dokumentasi di sini
    '''

    def __init__(self):
        
        # make connection to product.db
        # connp -> connection + product.db
        connp = sqlite3.connect('product.db')
        # cp -> cursor + product.db
        cp = connp.cursor()
        # enable foreign keys
        cp.execute('PRAGMA foreign_keys = ON')
        connp.commit()

        # make connection to validasi.db
        # connv -> connection + validasi.db
        connv = sqlite3.connect('validasi.db')
        # cv -> cursor + validasi.db
        cv = connv.cursor()
        # enable foreign keys
        cv.execute('PRAGMA foreign_keys = ON')
        connv.commit()

    def listID():
        '''
        '''
        pass


def main():
    pass


