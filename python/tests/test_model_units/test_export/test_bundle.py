# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from coreai_models.export.bundle import _get_build_info


def test_build_info_includes_core_packages():
    info = _get_build_info()
    assert "coreai-core" in info
    assert "coreai-torch" in info
    assert "coreai-opt" in info
    assert "torch" in info
    for version in info.values():
        assert version  # non-empty string
