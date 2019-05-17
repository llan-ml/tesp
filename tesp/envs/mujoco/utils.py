# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

ENV_ASSET_DIR = osp.join(osp.dirname(__file__), "assets")


def get_asset_xml(xml_name):
    return osp.join(ENV_ASSET_DIR, xml_name)
