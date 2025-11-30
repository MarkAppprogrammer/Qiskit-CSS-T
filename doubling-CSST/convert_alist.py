import numpy as np
import sys


def readAlist(directory):
    '''
    Reads in a parity check matrix (pcm) in A-list format from text file.

    Parameters
    ----------
    directory : string
        directory of text file of pcm to be read in.

    Returns
    -------
    alist_numpy : array
        returns the pcm in form of a numpy array with 0/1 bits as float64.
        
    '''
    alist_raw = []
    with open(directory, "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove trailing newline \n and split at spaces:
            line = line.rstrip().split()
            # map string to int:
            line = list(map(int, line))   
            alist_raw.append(line)
    alist_numpy = alistToNumpy(alist_raw)
    # alist_numpy = alist_numpy.astype(float)
    return alist_numpy


def alistToNumpy(lines):
    """
    Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
    list-of-lists corresponding to the lines of the AList format, already parsed to integers
    if read from a text file.
    The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.
    This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
    weights, respectively) and the row-based information (last part of the Alist file) are omitted.
    Example:
        >>> alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
        array([[1, 0, 1],
               [0, 1, 1]])
    """
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=int)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix

def strToAlistArray(alist_str):
    """Converts a string in MacKay's AList format into a list of lists of integers.
    """
    lines = alist_str.strip().split('\n')
    alist_array = []
    for line in lines:
        entries = list(map(int, line.strip().split()))
        alist_array.append(entries)
    return alist_array

def numpyToAlist(matrix):
    """Converts a 2-dimensional 0/1 numpy array into MacKay's AList format, in form of a list of
    lists of integers.
    """
    if sys.version_info[0] == 2:
        import cStringIO
        StringIO = cStringIO.StringIO
    else:
        import io
        StringIO = io.StringIO

    with StringIO() as output:
        nRows, nCols = matrix.shape
        # first line: matrix dimensions
        output.write('{} {}\n'.format(nCols, nRows))

        # next three lines: (max) column and row degrees
        colWeights = matrix.sum(axis=0)
        rowWeights = matrix.sum(axis=1)

        maxColWeight = max(colWeights)
        maxRowWeight = max(rowWeights)

        output.write('{} {}\n'.format(maxColWeight, maxRowWeight))
        output.write(' '.join(map(str, colWeights)) + '\n')
        output.write(' '.join(map(str, rowWeights)) + '\n')

        def writeNonzeros(rowOrColumn, maxDegree):
            nonzeroIndices = np.flatnonzero(rowOrColumn) + 1  # AList uses 1-based indexing
            output.write(' '.join(map(str, nonzeroIndices)))
            # fill with zeros so that every line has maxDegree number of entries
            output.write(' 0' * (maxDegree - len(nonzeroIndices)))
            output.write('\n')

        # column-wise nonzeros block
        for column in matrix.T:
            writeNonzeros(column, maxColWeight)
        for row in matrix:
            writeNonzeros(row, maxRowWeight)
        return output.getvalue()

