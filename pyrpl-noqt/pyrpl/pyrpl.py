###############################################################################
#    pyrpl - DSP servo controller for quantum optics with the RedPitaya
#    Copyright (C) 2014-2016  Leonhard Neuhaus  (neuhaus@spectro.jussieu.fr)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

""" # DEPRECATED DOCSTRING - KEEP UNTIL DOCUMENTATION IS READY
pyrpl.py - high-level lockbox functionality

A lockbox is a device that converts a number of input signals into a number of
output signals suitable to stabilize a physical system in a desired state. This
task is generally divided into two steps:
1) bring the system close the desired state where it can be linearized
2) keep it there using linear control.

The task is further divided into several subtasks:
0a) Condition the input signals so that they are suitable for the next steps
- offset removal
- input filters
- demodulation / lockin
- inversion
0b) Estimate the system state from the past and present history of input and
output signals.
0c) Build a filter for the output signals such that they can be conveniently
addressed with higher-level lockbox logic.

1) As above: apply reasonable action to the outputs to approach the system to
the desired state. We generally call this the 'search' step.
- provide a number of algorithms/recipes to do this

2) As above: Turn on linear feedback. We call this the 'lock' step.
- connect the input signals with appropriate gain to the outputs
- the gain depends on the state of the system, so internal state representation
will remain useful here

This naturally divides the lockbox object into 3 subcomponents:
a) inputs
b) internal model
c) outputs

which will be interconnected by the algorithms that come with the model and
make optimal use of the available inputs and outputs. The job of the
configuration file is to provide a maximum of information about the inputs,
outputs and the physical system (=internal model) so that the lockbox is
effective and robust. The lockbox will usually require both a coarse-tuning
and an optimization step for optimum performance, which will both adjust the
various parameters for the best match between model and real system.

Let's make this reasoning more clear with an example:

A Fabry-Perot cavity is to be locked near resonance using a PDH scheme. The
incident laser beam goes through a phase modulator. The cavity contains a piezo
with estimated bandwidth 10 kHz (appearance of first resonance) and
a displacement of 350 nm/V that goes into the piezo amplifier. To limit the
effect of amplifier noise, we have inserted an RC lowpass between amplifier and
piezo with a cutoff frequency of 100 Hz. The laser contains another piezo with
estimated bandwidth of 50 kHz that changes the laser frequency by 5 MHz/V. An
RC filter provides a lowpass with 1kHz cutoff. Finally, the cavity can be tuned
through its temperature with a bandwidth slower than 0.1 Hz. We estimate from
thermal expansion coefficients that 1 V to the Peltier amplifier leading to 3 K
heating of the cavity spacer should lead to 30ppm/K*20cm*3K/V = 18 micron/V
length change. Both reflection and transmission of the cavity are available
error signals. The finesse of the cavity is 5000, and therefore there are
large regions of output signals where no useful error signal can be obtained.

We first generate a clean list of available inputs and outputs and some
parameters of the cavity that we know already:

inputs:
  in1:
    reflection
  in2:
    transmission
  # also possible
  # in2: pdh # for externally generated pdh
outputs:
  out1:
    # we insert a bias-T with separation frequency around 1 MHz behind out1
    # this allows us to use the fast output for both the piezo and PDH
    modulator:
      amplitude: 0.1
      frequency: 50e6
    cavitypiezo:
      # piezo specification: 7 micron/1000V
      # amplifier gain: 50
      # therefore effective DC gain: 350nm/V
      m_per_V: 350e-9
      bandwidth: 100.0
  out2:
    laserpiezo:
      Hz_per_V: 5e6
      bandwidth: 1e3
  pwm1:
    temperature:
      m_per_V: 18e-6
      bandwidth: 0.1
model:
  type: fabryperot
  wavelength: 1064e-9
  finesse: 5000
  # round-trip length in m (= twice the length for ordinary Fabry-Perot)
  length: 0.72
  lock: # lock methods in order of preferrence
    order:
      pdh
      reflection
      transmission
    # when approaching a resonance, we can either abruptly jump or smoothly
    # ramp from one error signal to another. We specify our preferrence with
    # the order of keywords after transition
    transition: [ramp, jump]
    # target value for our lock. The API provides many ways to adjust this at
    # runtime
    target:
      detuning: 0
  # search algorithms to use in order of preferrence, as available in model
  search:
    drift
    bounce

Having selected fabryperot as modeltype, the code will automatically search
for a class named fabryperot in the file model.py to provide for the internal
state representation and all algorithms. You can create your own model by
adding other classes to this file, or by inheriting from existing ones and
adding further functionality. The naming of all other configuration parameters
is linked to the model, since all functionality that makes use of these para-
meters is implemented there. Another very often used model type is
"interferometer". The only difference is here that

"""

from __future__ import print_function

import logging
import os
import os.path as osp
from shutil import copyfile

from .memory import MemoryTree
from .redpitaya import RedPitaya
from . import pyrpl_utils

# loaded when we get started
from . import user_config_dir
from ._version import __version__

# input is the wrong function in python 2
try:
    raw_input
except NameError:  # Python 3
    raw_input = input

try:
    basestring  # in python 2
except:
    basestring = (str, bytes)


default_pyrpl_config = {'name': 'default_pyrpl_instance',
                        'loglevel': 'info',
                        'background_color': '',
                        # reasonable options:
                        # 'CCCCEE',  # blueish
                        # 'EECCCC', # reddish
                        # 'CCEECC', # greenish
                        'modules': ['NetworkAnalyzer',
                                    'SpectrumAnalyzer',
                                    'CurveViewer',
                                    'PyrplConfig',
                                    'Lockbox'
                                    ]}

help_message = """
PyRPL version %s command-line help
==================================

Syntax for launching PyRPL
------------------------------------------------------------------------------
Rectangular brackets [] indicate optional parameters.

Syntax for binary executable:
    pyrpl [key1=value1 [key2=value2 [key3=value3 [...]]]]

Syntax with python installation:
    python -m pyrpl [key1=value1 [key2=value2 [key3=value3 [...]]]]

Syntax from within Python:
    from pyrpl import Pyrpl
    p = Pyrpl([key1=value1, [key2=value2 [key3=value3 [...]]]])


Keys and Values:
------------------------------------------------------------------------------
config   configuration name (without .yml-extension)
source   name of the configuration to use as default

hostname hostname of the redpitaya
user     username for ssh login on the redpitaya
password password for ssh login on the redpitaya
sshport  port for ssh, default is 22
port     port for redpitaya_client, default is 2222

gui      one of [True, False], to en- or disable GUI
loglevel logging level, one of [debug, info, warning, error]
"""%(__version__)


class Pyrpl(object):
    """
    Higher level object, in charge of loading the right hardware and software
    module, depending on the configuration described in a config file.

    Parameters
    ----------
    config: str
        Name of the config file. No .yml extension is needed. The file
        should be located in the config directory.
    source: str
        If None, it is ignored. Else, the file 'source' is taken as a
        template config file and copied to 'config' if that file does
        not exist.
    **kwargs: dict
        Additional arguments can be passed and will be written to the
        redpitaya branch of the config file. See class definition of
        RedPitaya for possible keywords.
    """
    def __init__(self,
                 config=None,
                 source=None,
                 **kwargs):
        # logger initialisation
        self.logger = logging.getLogger(name='pyrpl') # default: __name__
        # configuration is retrieved from config file
        self.c = MemoryTree(filename=config, source=source)
        if self.c._filename is not None:
            self.logger.info("All your PyRPL settings will be saved to the "
                             "config file\n"
                             "    %s\n"
                             "If you would like to restart "
                             "PyRPL with these settings, type \"pyrpl.exe "
                             "%s\" in a windows terminal or \n"
                             "    from pyrpl import Pyrpl\n"
                             "    p = Pyrpl('%s')\n"
                             "in a python terminal.",
                             self.c._filename,
                             self.c._filename_stripped,
                             self.c._filename_stripped)
        # make sure config file has the required sections and complete with
        # missing entries from default
        pyrplbranch = self.c._get_or_create('pyrpl')
        for k in default_pyrpl_config:
            if k not in pyrplbranch._keys():
                if k =='name':
                    # assign the same name as in config file by default
                    pyrplbranch[k] = self.c._filename_stripped
                else:
                    # all other (static) defaults
                    pyrplbranch[k] = default_pyrpl_config[k]
        # set global logging level if specified in kwargs or config file
        if 'loglevel' in kwargs:
            self.c.pyrpl.loglevel = kwargs.pop('loglevel')
        pyrpl_utils.setloglevel(level=self.c.pyrpl.loglevel,
                                loggername='pyrpl')
        # initialize RedPitaya object with the configured or default parameters
        self.c._get_or_create('redpitaya')
        self.c.redpitaya._update(kwargs)
        self.name = pyrplbranch.name
        self.rp = RedPitaya(config=self.c)
        self.redpitaya = self.rp  # alias
        self.rp.parent=self

    @property
    def hardware_modules(self):
        """
        List of all hardware modules loaded in this configuration.
        """
        if self.rp is not None:
            return list(self.rp.modules.values())
        else:
            return []

    def _clear(self):
        """
        kill all timers and closes the connection to the redpitaya
        """
        # make sure the save timer of the config file is not running and
        # all data are written to the harddisk
        self.c._write_to_file()
        # end redpitatya communication
        self.rp.end_all()
