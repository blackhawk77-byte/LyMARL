import numpy as np
from typing import Tuple
from collections import deque
from collections import defaultdict



class UserEquipment: 
    """ 
    A class to represent a User Equipment (UE)
    includes all necessary attributes and methods for UE operations.
    """
    def __init__(self, ue_id, position): 
        """ 
        Initialize the UE. 
        
        Paramaters: 
        - ue_id (int): Unique identifier for the UE 
        - position (tuple): Position of the UE in Cartesian coordinates (x, y)
        """
        self.ue_id = ue_id 
        self.position = np.array(position) 
        self.snr_list = {}  # Dictionary: key=BS id, value = SNR in dB 

    def add_snr(self, bs_id, snr): 
        """ 
        Add or update the SNR value from a specific base station
        
        Parameters: 
        - bs_id (int): ID of the bawse station
        - snr (float): Signal-to-Noise Ratio in dB
        """
        self.snr_list[bs_id] = snr 

    def best_bs(self): 
        """ 
        Return the base station ID with the maximum SNR, besides the macro base station
        """
        if not self.snr_list: 
            return None 
        # Exclude MBS (bs_id=0) from the best BS selection 
        candidates = {k: v for k, v in self.snr_list.items() if k != 0}

        return max(candidates, key=candidates.get) if candidates else None
    
    def get_snr(self, bs_id): 
        """ 
        Retrieve SNR for a given BS. 
        
        Parameters: 
        - bs_id (int): ID of the base station 
        
        Returns: 
        - float: SNR value in dB or None if not availbale 
        """
        return self.snr_list.get(bs_id, -np.inf) 

    def __repr__(self):
        return f"UE#{self.ue_id} | [x, y] = {self.position} | Best BS: {self.best_bs()}"
    

