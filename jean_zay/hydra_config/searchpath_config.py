# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf

def qos_from_hours(hours):
    if hours > 20:
        qos = 't4'
    elif hours > 2:
        qos = 't3'
    else:
        qos = 'dev'
    return qos

OmegaConf.register_new_resolver('multiply10', lambda x: x * 10)
OmegaConf.register_new_resolver('qos_from_hours', qos_from_hours)

class JeanZaySearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(
            provider="submission-scripts", path="pkg://jean_zay/hydra_config"
        )