from common import load_csv_as_dict, write_rows_to_csv
import os
import pathlib


def align_exact_matches(oiv_classes_path, aligned_autotags_path):
    """ Iterates over class names in Open Images V5, and if there is an exact match in name with a YFCC100 class name,
        that class is labeled as corresponding to the classes Open Images label

    Parameters
    ----------
    oiv_classes_path : str
        Path to oiv_classes file, which is list oo boxable classes in rows of format "/m/011k07,Tortoise", where the
        left item is the oiv label and right item is the corresponding oiv name. Typically file
        "class-descriptions-boxable.csv" called Class Names on OIV website
    aligned_autotags_path : str
        Path to file mapping YFCC100M auto tag names to OIV labels. Should be CSV of rows of format "chipmunk,/m/04rky"
        where left item is YFCC100M auto tag name, and right item is the corresponding OIV label, if no OIV label is
        assigned yet the right item should be 0
    """

    oiv_classes_csv = load_csv_as_dict(oiv_classes_path, fieldnames=["oiv_label", "oiv_name"], delimiter=",")
    oiv_label_map = {row["oiv_name"].lower(): row["oiv_label"] for row in oiv_classes_csv}
    yfcc_100m_labels = load_csv_as_dict(aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    new_rows = []
    i = 0
    for row in yfcc_100m_labels:
        yfcc100m_name, oiv_labels = row["yfcc100m_name"].lower(), row["oiv_label"]
        if oiv_labels == '0' and oiv_label_map.get(yfcc100m_name):
            row["oiv_label"] = oiv_label_map[yfcc100m_name]
            i += 1
        new_rows.append(row)
    write_rows_to_csv(new_rows, aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    print("Found {} exact matches".format(i))


def find_unaligned_classes(oiv_classes_path, aligned_autotags_path):
    """ Iterates over YFCC100 class names, and finds the ones which have not been assigned a corresponding OIV label,
        returns the dict OIV labels to OIV names for OIV labels which have not been assigned

    Parameters
    ----------
    oiv_classes_path : str
        Path to oiv_classes file, which is list oo boxable classes in rows of format "/m/011k07,Tortoise", where the
        left item is the oiv label and right item is the corresponding oiv name. Typically file
        "class-descriptions-boxable.csv" called Class Names on OIV website
    aligned_autotags_path : str
        Path to file mapping YFCC100M auto tag names to OIV labels. Should be CSV of rows of format "chipmunk,/m/04rky"
        where left item is YFCC100M auto tag name, and right item is the corresponding OIV label, if no OIV label is
        assigned yet the right item should be 0

    Returns
    -------
    dict of str -> str
        Maps OIV labels to OIV names for OIV labels which have not been assigned
    """

    oiv_classes_csv = load_csv_as_dict(oiv_classes_path, fieldnames=["oiv_label", "oiv_name"], delimiter=",")
    oiv_name_map = {row["oiv_label"]: row["oiv_name"].lower() for row in oiv_classes_csv}
    yfcc_100m_labels = load_csv_as_dict(aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    for row in yfcc_100m_labels:
        yfcc100m_name, oiv_labels = row["yfcc100m_name"].lower(), row["oiv_label"]
        for oiv_label in oiv_labels.split("&"):
            if oiv_name_map.get(oiv_label):
                del oiv_name_map[oiv_label]
    return oiv_name_map


def find_oiv_class_alignments(oiv_classes_path, aligned_autotags_path):
    """ Returns a dict of OIV names as keys with each having a list of YFCC100M names that have been assigned to them

    Parameters
    ----------
    oiv_classes_path : str
        Path to oiv_classes file, which is list oo boxable classes in rows of format "/m/011k07,Tortoise", where the
        left item is the oiv label and right item is the corresponding oiv name. Typically file
        "class-descriptions-boxable.csv" called Class Names on OIV website
    aligned_autotags_path : str
        Path to file mapping YFCC100M auto tag names to OIV labels. Should be CSV of rows of format "chipmunk,/m/04rky"
        where left item is YFCC100M auto tag name, and right item is the corresponding OIV label, if no OIV label is
        assigned yet the right item should be 0

    Returns
    -------
    dict of str -> list of str
        Maps OIV names to the corresponding YFCC100M names that have been assigned to them
    """

    oiv_classes_csv = load_csv_as_dict(oiv_classes_path, fieldnames=["oiv_label", "oiv_name"], delimiter=",")
    oiv_name_map = {row["oiv_label"]: row["oiv_name"].lower() for row in oiv_classes_csv}
    yfcc_100m_labels = load_csv_as_dict(aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    oiv_yfcc100m_map = {}
    for row in yfcc_100m_labels:
        yfcc100m_name, oiv_labels = row["yfcc100m_name"].lower(), row["oiv_label"]
        if oiv_labels != '0':
            for oiv_label in oiv_labels.split("&"):
                oiv_name = oiv_name_map[oiv_label]
                if not oiv_yfcc100m_map.get(oiv_name):
                    oiv_yfcc100m_map[oiv_name] = []
                oiv_yfcc100m_map[oiv_name].append(yfcc100m_name)
    return oiv_yfcc100m_map


def get_yfcc100m_oiv_labels_map():
    """ Returns a dict of YFCC100M names as keys with the value being their corresponding oiv_labels

    Returns
    -------
    dict of str -> str
        Maps YFCC100M names to the corresponding OIV label that have been assigned to them
    """
    aligned_autotags_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "aligned_autotags.txt")
    yfcc_100m_labels = load_csv_as_dict(aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    yfcc_100m_name_oiv_label_map = {row["yfcc100m_name"]: row["oiv_label"] for row in yfcc_100m_labels if
                                    row["oiv_label"] != "0"}
    return yfcc_100m_name_oiv_label_map


def replace_oiv_name_with_label(oiv_classes_path, aligned_autotags_path):
    """ Replaces OIV names in the OIV label column of aligned_autotags with the corresponding OIV label

    Parameters
    ----------
    oiv_classes_path : str
        Path to oiv_classes file, which is list oo boxable classes in rows of format "/m/011k07,Tortoise", where the
        left item is the oiv label and right item is the corresponding oiv name. Typically file
        "class-descriptions-boxable.csv" called Class Names on OIV website
    aligned_autotags_path : str
        Path to file mapping YFCC100M auto tag names to OIV labels. Should be CSV of rows of format "chipmunk,/m/04rky"
        where left item is YFCC100M auto tag name, and right item is the corresponding OIV label, if no OIV label is
        assigned yet the right item should be 0. For this method OIV label map be the OIV name, in which case it will
        be replaced with the corresponding OIV label after the method is run
    """

    yfcc_100m_labels = load_csv_as_dict(aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
    oiv_classes_csv = load_csv_as_dict(oiv_classes_path, fieldnames=["oiv_label", "oiv_name"], delimiter=",")
    oiv_label_map = {row["oiv_name"]: row["oiv_label"] for row in oiv_classes_csv}
    oiv_labels = set(oiv_label_map.values())
    new_rows = []
    for row in yfcc_100m_labels:
        oiv_label = row["oiv_label"]
        if oiv_label != "0":
            if not oiv_labels.get(oiv_label):
                row["oiv_label"] = oiv_label_map[row["oiv_label"]]
        new_rows.append(row)
    write_rows_to_csv(new_rows, aligned_autotags_path, fieldnames=["yfcc100m_name", "oiv_label"], delimiter=",")
