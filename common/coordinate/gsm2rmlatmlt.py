import numpy as np


def _gsm2rmlatmlt(x_gsm, y_gsm, z_gsm):
    """
    Converts GSM (Geocentric Solar Magnetospheric) coordinates to 
    R (radial distance), MLAT (Magnetic Latitude), and MLT (Magnetic Local Time).
    
    Parameters:
    -----------
    x_gsm, y_gsm, z_gsm : float or np.ndarray
        Coordinates in GSM system. If input is in km, R will be in km.
        If input is in Re (Earth Radii), R will be in Re.

    Returns:
    --------
    r : float or np.ndarray
        Radial distance from the center of the Earth.
    mlat : float or np.ndarray
        Magnetic Latitude in degrees [-90, 90].
    mlt : float or np.ndarray
        Magnetic Local Time in decimal hours [0, 24).
    """
    
    # 1. Calculate Radial Distance (R)
    # The magnitude of the vector in any orthogonal coordinate system.
    r = np.sqrt(x_gsm**2 + y_gsm**2 + z_gsm**2)
    
    # 2. Calculate Magnetic Latitude (MLAT)
    # In GSM, the Z-axis is the projection of the dipole axis.
    # MLAT is the angle from the XY-plane towards the Z-axis.
    # Handle potential division by zero if r=0
    mlat = np.rad2deg(np.arcsin(np.clip(z_gsm / r, -1.0, 1.0)))
    
    # 3. Calculate Magnetic Local Time (MLT)
    # The angle in the GSM XY-plane.
    # X_gsm points to the Sun (MLT 12).
    # Y_gsm points towards dusk (MLT 18).
    
    # atan2(y, x) returns angle in radians between [-pi, pi]
    phi_rad = np.arctan2(y_gsm, x_gsm)
    phi_deg = np.rad2deg(phi_rad)
    
    # Convert angle to MLT hours:
    # 0 deg (Sunward) -> 12 MLT
    # 90 deg (Duskward) -> 18 MLT
    # -90 deg (Dawnward) -> 6 MLT
    # 180/-180 deg (Tailward) -> 0 MLT
    mlt = (phi_deg / 15.0) + 12.0
    
    # Wrap MLT to be within [0, 24)
    mlt = np.mod(mlt, 24.0)
    
    return r, mlat, mlt


def _rmlatmlt2gsm(r, mlat, mlt):
        """
        Converts R, MLAT, and MLT back to GSM coordinates.
        
        Parameters:
            r    : Radial distance
            mlat : Magnetic Latitude (degrees)
            mlt  : Magnetic Local Time (decimal hours)
            
        Returns:
            x_gsm, y_gsm, z_gsm
        """
        # Convert MLT back to angle phi (degrees) in XY plane
        # 12 MLT -> 0 deg, 18 MLT -> 90 deg, 0 MLT -> -180 deg
        phi_deg = (mlt - 12.0) * 15.0
        
        phi_rad = np.deg2rad(phi_deg)
        mlat_rad = np.deg2rad(mlat)
        
        # Projection logic:
        # z = r * sin(mlat)
        # xy_proj = r * cos(mlat)
        # x = xy_proj * cos(phi)
        # y = xy_proj * sin(phi)
        
        xy_proj = r * np.cos(mlat_rad)
        
        x_gsm = xy_proj * np.cos(phi_rad)
        y_gsm = xy_proj * np.sin(phi_rad)
        z_gsm = r * np.sin(mlat_rad)
        
        return x_gsm, y_gsm, z_gsm
