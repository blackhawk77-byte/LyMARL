import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from collections import deque 

class BaseStation: 
    """ 
    Base calss representing a general base station. 
    Includes basic attributes like position, transmit power, frequency, and bandwidth. (or config_dict) 
    """

    def __init__(self,
                 bs_id: int, 
                 position: Tuple[float, float],
                 tx_power_dbm: float = None, 
                 frequency: float = None,
                 antenna_gain_tx: float = None, 
                 antenna_gain_rx: float = None, 
                 bandwidth: float = None, 
                 shadowing_std_dev: float = None, 
                 path_loss_exp: float = None, 
                 reference_distance: float = None, 
                 noise_figure_db: float = None, 
                 beam_limit: int = np.inf,
                 config_dict: dict = None):
        if config_dict:
            self.bs_id = bs_id
            self.position = config_dict.get('position', position)
            self.tx_power_dbm = config_dict.get('tx_power_dbm', tx_power_dbm)
            self.frequency = config_dict.get('frequency_mhz', frequency)
            self.antenna_gain_tx = config_dict.get('antenna_gain_tx', antenna_gain_tx)
            self.antenna_gain_rx = config_dict.get('antenna_gain_rx', antenna_gain_rx)
            self.bandwidth = config_dict.get('bandwidth_mhz', bandwidth)
            self.shadowing_std_dev = config_dict.get('shadowing_std_dev', shadowing_std_dev)
            self.path_loss_exp = config_dict.get('path_loss_exp', path_loss_exp)
            self.reference_distance = config_dict.get('reference_distance', reference_distance)
            self.noise_figure_db = config_dict.get('noise_figure_db', noise_figure_db)
            self.beam_limit = config_dict.get('beam_limit', beam_limit)
        else:
            self.bs_id = bs_id
            self.position = position
            self.tx_power_dbm = tx_power_dbm
            self.frequency = frequency
            self.antenna_gain_tx = antenna_gain_tx
            self.antenna_gain_rx = antenna_gain_rx
            self.bandwidth = bandwidth
            self.shadowing_std_dev = shadowing_std_dev
            self.path_loss_exp = path_loss_exp
            self.reference_distance = reference_distance
            self.noise_figure_db = noise_figure_db
            self.beam_limit = beam_limit
        self.wavelength = 3e8 / (self.frequency)  # Convert MHz to Hz
        self.connected_users = [] # List to store connected users 



    def distance_to(self, user_pos) -> float: 
        """ 
        Calculate the Euclidean distance to a user position. 
        
        Args: 
            user_pos (np.ndarray): User position as a numpy array.
        
        Returns:
            distance (float): Euclidean distance to the user."""
        
        return np.linalg.norm(np.array(user_pos) - np.array(self.position)) 
    
    def path_loss(self, distance): 
        """ 
        Calculate the path loss based on the close-in free space model. 
        This method can be overriden by subclasses for different path loss models. 
        Args: 
            distance (float): Distance to the user in meters.
            
        Returns: 
            path_loss_db (float): Path loss in dB.
        """
        if self.reference_distance is None: 
            raise ValueError("Reference distance must be set for path loss calculation.") 
        fspl_db = 20 * np.log10(4 * np.pi * self.reference_distance / self.wavelength)
        pl_db = fspl_db + 10 * self.path_loss_exp * np.log10(distance / self.reference_distance) 
        shadowing = np.random.normal(0, self.shadowing_std_dev) if self.shadowing_std_dev else 0
        return pl_db + shadowing 
    
    def receive_power(self, distance) -> float: 
        """ 
        Calculate the received power at the user based on the path loss. 
        
        Args: 
            distance (float): Distance to the user in meters.
        
        Returns: 
            received_power_dbm (float): Received power in dBm.
        """
        if self.tx_power_dbm is None:
            raise ValueError("Transmit power must be set for receive power calculation.")
        
        path_loss_db = self.path_loss(distance)
        received_power_dbm = self.tx_power_dbm + self.antenna_gain_tx + self.antenna_gain_rx - path_loss_db
        return received_power_dbm
    
    def can_serve(self, user_pos): 
        """ 
        Default to serve every user within the beam limit, unless overridden.
        
        Args: 
            user_pos (np.ndarray): User position as a numpy array.
        
        Returns: 
            can_serve (bool): True if the user is within the beam limit, False otherwise.
        """
        return True 
    
    def reset(self): 
        """ 
        Reset the connected users list. 
        """
        self.connected_users = []
    



class MacroBaseStation(BaseStation): 
    """
    Represents a macro base station with full coverage of the area. 
    """
    def __init__(self, bs_id, position): 
        super().__init__(bs_id=bs_id, 
                         position=position, 
                         tx_power_dbm=46, 
                         frequency=2e9, 
                         antenna_gain_tx=17, 
                         antenna_gain_rx=0,
                         bandwidth=10e6, 
                         shadowing_std_dev=np.sqrt(9), 
                         path_loss_exp=3.76,
                         reference_distance=20.7, 
                         noise_figure_db=5,
                         beam_limit=np.inf)
        
    def can_serve(self, user_pos): 
        return True 
    

class SmallCellBaseStation(BaseStation): 
    """ 
    Represents a small cell base station with liimited coverage radius (default=35m) 
    """
    def __init__(self, bs_id, position, beam_limit, coverage_radius: float = 35): 
        super().__init__(bs_id=bs_id, 
                         position=position, 
                         tx_power_dbm=20, 
                         frequency=28e9,            # 28 GHz frequency for small cells
                         antenna_gain_tx=0,         # Tx Antenna gain = 0dB (1.0)
                         antenna_gain_rx=0,         # Rx Antenna gain = 0dB (1.0) 
                         bandwidth=500e6,           # Bandwidth = 500MHz
                         shadowing_std_dev=np.sqrt(12), 
                         path_loss_exp=2.5,         # Path-loss coefficient 
                         reference_distance=5,      # Reference distance 
                         noise_figure_db=0,         
                         beam_limit=beam_limit, )
        self.coverage_radius = coverage_radius

    def can_serve(self, user_pos): 
        """Smalle cell can only serve users within its coverage radius. """
        ##d = self.distance_to(user_pos)
        return True
    
    def path_loss(self, distance): 
        """ 
        Calculate the path loss for small cells using a different model. According to the paper 
        :param distance(float): Distance to the user in meters 
        :return : path loss in dB 
        """
        return 128.1 + 37.6 * np.log10(distance / 1000) 

