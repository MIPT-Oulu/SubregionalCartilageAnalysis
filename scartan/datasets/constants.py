from collections import OrderedDict


def invert_mappings(d_in):
    d_out = OrderedDict()
    for k, vs in d_in.items():
        for v in vs:
            d_out.setdefault(v, []).append(k)
    return d_out


"""
OAI iMorphics reference classes (DICOM attribute names). Cartilage tissues
"""
locations_mh53 = OrderedDict([
    ('Background', 0),
    ('FemoralCartilage', 1),
    ('LateralTibialCartilage', 2),
    ('MedialTibialCartilage', 3),
    ('PatellarCartilage', 4),
    ('LateralMeniscus', 5),
    ('MedialMeniscus', 6),
])


"""
Segmentation predictions. Major cartilage tissues. Joined L and M
"""
locations_f43h = OrderedDict([
    ('_background', 0),
    ('femoral', 1),
    ('tibial', 2),
])


"""
Segmentation predictions. Cartilage tissues. Joined L and M
"""
locations_zp3n = OrderedDict([
    ('_background', 0),
    ('femoral', 1),
    ('tibial', 2),
    ('patellar', 3),
    ('menisci', 4),
])


"""
Biomediq. Compartments of the cartilage tissues
"""
locations_61om = OrderedDict([
    ('_background', 0),
    ('medial_femoral', 1),
    ('lateral_femoral', 2),
    ('medial_tibial', 3),
    ('lateral_tibial', 4),
    ('patellar', 5),
    ('medial_meniscus', 6),
    ('lateral_meniscus', 7),
])


"""
Chondrometrics. Projects 22, 66, etc (75%). Compartments of the cartilage tissues
See: Wirth W., Eckstein F. A technique for regional analysis of femorotibial cartilage
     thickness based on quantitative magnetic resonance imaging.
     IEEE Trans Med Imaging. 2008 Jun;27(6):737-44.
     http://dx.doi.org/10.1109/TMI.2007.907323 (4)
"""
locations_w02n = OrderedDict([
    ('_background', 0),
    ('tLF', 1),  # lateral femoral
    ('ccLF', 2),
    ('ecLF', 3),
    ('icLF', 4),
    ('pLF', 5),
    ('tMF', 6),  # medial femoral
    ('ccMF', 7),
    ('ecMF', 8),
    ('icMF', 9),
    ('pMF', 10),
    ('aLT', 11),  # lateral tibial
    ('pLT', 12),
    ('cLT', 13),
    ('eLT', 14),
    ('iLT', 15),
    ('aMT', 16),  # medial tibial
    ('pMT', 17),
    ('cMT', 18),
    ('eMT', 19),
    ('iMT', 20),
    ('patellar', 21),
    ('medial_meniscus', 22),
    ('lateral_meniscus', 23),
])


# Segm <-> iMorphics
valid_mappings_zp3n_mh53 = OrderedDict([
    (0, (0, )),
    (1, (1, )),
    (2, (2, 3)),
    (3, (4, )),
    (4, (5, 6)),
])
valid_mappings_mh53_zp3n = invert_mappings(valid_mappings_zp3n_mh53)


# Segm <-> Biomediq
valid_mappings_zp3n_61om = OrderedDict([
    (0, (0,)),
    (1, (1, 2)),
    (2, (3, 4)),
    (3, (5,)),
    (4, (6, 7)),
])
valid_mappings_61om_zp3n = invert_mappings(valid_mappings_zp3n_61om)


# iMorphics <-> Biomediq
valid_mappings_mh53_61om = OrderedDict([
    (0, (0,)),
    (1, (1, 2)),
    (2, (4,)),
    (3, (3,)),
    (4, (5,)),
    (5, (7,)),
    (6, (6,)),
])
valid_mappings_61om_mh53 = invert_mappings(valid_mappings_mh53_61om)


# Segm <-> Chondrometrics
valid_mappings_zp3n_w02n = OrderedDict([
    (0, (0,)),
    (1, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    (2, (11, 12, 13, 14, 15, 16, 17, 18, 19, 20)),
    (3, (21,)),
    (4, (22, 23)),
])
valid_mappings_w02n_zp3n = invert_mappings(valid_mappings_zp3n_w02n)


# iMorphics <-> Chondrometrics
valid_mappings_mh53_w02n = OrderedDict([
    (0, (0,)),
    (1, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    (2, (11, 12, 13, 14, 15)),
    (3, (16, 17, 18, 19, 20)),
    (4, (21,)),
    (5, (23,)),
    (6, (22,)),
])
valid_mappings_w02n_mh53 = invert_mappings(valid_mappings_mh53_w02n)


atlas_to_locations = {
    'segm': locations_zp3n,
    'imo': locations_mh53,
    'biomediq': locations_61om,
    'chondr75n': locations_w02n,
}
