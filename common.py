import csv

import sys

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def load_csv_as_dict(csv_path, fieldnames=None, delimiter=None):
    """ Loads the csv DictReader

    Parameters
    ----------
    csv_path : str
        Path to csv
    fieldnames : list of str
        List of fieldnames, if None then fieldnames are take from the first row
    delimiter : str
        Delimiter to split on, default \t

    Returns
    -------
    csv.DictReader
        DictReader object of path
    """

    delimiter = delimiter or "\t"
    f = open(csv_path, encoding='utf8')
    c = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter)
    return c


def write_rows_to_csv(rows_to_write, csv_path, fieldnames=None, mode=None):
    """ Write the rows the csv at the path

    Parameters
    ----------
    rows_to_write : list of dict
        Rows to write to the CSV
    csv_path : str
        Path to csv
    fieldnames : list of str
        List of fieldnames for csv. Default of keys of first row in rows_to_write
    mode : str
        Mode to write to file (w for write, a for append)

    Returns
    -------
    csv.DictWriter
        DictWriter object of path
    """

    if rows_to_write:
        mode = mode or "w"
        fieldnames = fieldnames or list(rows_to_write[0].keys())
        f = open(csv_path, mode, encoding='utf8', newline='')
        c = csv.DictWriter(f, fieldnames=fieldnames)
        c.writerows(rows_to_write)
        f.close()
