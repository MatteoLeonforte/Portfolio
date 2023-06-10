from const import EstimatorConstant
import numpy as np
from numpy.random import uniform, multivariate_normal
from typing import Tuple


class EKF:
    """
    Extended Kalman Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
    """

    def __init__(
            self,
            estimator_constant: EstimatorConstant,
    ):
        self.constant = estimator_constant

    def initialize(
            self, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with the mean and covariance of the initial
        estimate.

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The mean of the initial state estimate. The order of states is
                given by x = [p_x, p_y, psi, tau, l].
            Pm : np.ndarray, dim: (num_states, num_states)
                The covariance of the initial state estimate. The order of
                states is given by x = [p_x, p_y, psi, tau, l].
        """

        self.L = offlineMatrix_L(self.constant.Ts)
        self.Q = np.diag([self.constant.sigma_beta**2, self.constant.sigma_uc**2])
        self.LQL = self.L @ self.Q @ self.L.transpose()
        self.R = np.diag([self.constant.sigma_GPS**2, self.constant.sigma_GPS**2, self.constant.sigma_psi**2, self.constant.sigma_tau**2])
        self.R_noGPS = np.diag([self.constant.sigma_psi**2, self.constant.sigma_tau**2])
        self.H = np.concatenate([np.eye(4,4),np.zeros([4,1])],axis=1)
        self.H_noGPS = self.H[2:4,:]
        self.M = np.eye(4,4)
        self.M_noGPS = self.M[2:4,2:4]
        self.MRM_noGPS = self.M_noGPS @ self.R_noGPS @ self.M_noGPS.transpose()
        self.MRM = self.M @ self.R @ self.M.transpose()

        tau_bar = self.constant.start_velocity_bound
        psi_bar = self.constant.start_heading_bound
        R = self.constant.start_radius_bound

        l_lb = self.constant.l_lb
        l_ub = self.constant.l_ub

        xm = [0,0,0,tau_bar/2,(l_lb+l_ub)/2]
        Pm = np.diag([np.square(R)/6,np.square(R)/6,np.square(2*psi_bar)/12,np.square(tau_bar)/12,np.square(l_ub-l_lb)/12])

        return xm, Pm

    def estimate(
            self,
            xm_prev: np.ndarray,
            Pm_prev: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the state of the vehicle.

        Args:
            xm_prev : np.ndarray, dim: (num_states,)
                The mean of the previous posterior state estimate xm(k-1). The
                order of states is given by x = [p_x, p_y, psi, tau, l].
            Pm_prev : np.ndarray, dim: (num_states, num_states)
                The covariance of the previous posterior state estimate Pm(k-1).
                The order of states is given by x = [p_x, p_y, psi, tau, l].
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [z_px, z_py, z_psi, z_tau].

        Returns:
            xm : np.ndarray, dim: (num_states,)
                The mean of the posterior estimate xm(k). The order of states is
                given by x = [p_x, p_y, psi, tau, l].
            Pm : np.ndarray, dim: (num_states, num_states)
                The covariance of the posterior estimate Pm(k). The order of
                states is given by x = [p_x, p_y, psi, tau, l].
        """

        no_GPS = False

        if not np.all(np.logical_not(np.isnan(measurement))):
            no_GPS = True


        # PRIOR UPDATE
        A_prev = onlineMatrix_A(Ts = self.constant.Ts, psi=xm_prev[2],tau = xm_prev[3],beta = np.arctan(0.5*np.tan(inputs[0])), l = xm_prev[4])
        xp = process_model_EKF(const=self.constant, xm_prev=xm_prev, input=inputs)
        Pp = A_prev @ Pm_prev @ A_prev.transpose() + self.LQL

        # MEASUREMENT UPDATE
        if no_GPS:
            K =   Pp @ self.H_noGPS.transpose()  @  np.linalg.inv( self.H_noGPS @ Pp @ self.H_noGPS.transpose() + self.MRM_noGPS)
            pred_error = measurement[2:] - xp[2:4]
            xm = xp + K @ (pred_error)
            Pm = (np.eye(5)- K @ self.H_noGPS) @ Pp
           
        else:
            K =   Pp @ self.H.transpose() @  np.linalg.inv( self.H @ Pp @ self.H.transpose() + self.MRM)
            pred_error = (measurement - xp[0:4])
            xm = xp + K @ (pred_error)
            Pm = (np.eye(5)- K @ self.H) @ Pp

        

        return xm, Pm


class PF:
    """
    Particle Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
        noise : str
            Type of noise "Non-Gaussian".
    """
    def __init__(
            self,
            estimator_constant: EstimatorConstant,
            noise: str,
    ):

        if noise == "Non-Gaussian":
            self.constant = estimator_constant
            self.noise = noise
            self.psi_bound = np.sqrt(3)*estimator_constant.sigma_psi
            self.tau_bound = np.sqrt(3)*estimator_constant.sigma_tau
            self.process_noise_bound = np.sqrt(3)*np.array([estimator_constant.sigma_beta,estimator_constant.sigma_uc])
            self.num_particles = 2500

            # PW variables
            norm_wp = 1/(estimator_constant.sigma_GPS*np.sqrt(2*np.pi))
            var_wp1 = -2/(estimator_constant.sigma_GPS**2)
            var_wp2 = 0.5*np.sqrt(3)*estimator_constant.sigma_GPS
            self.wp_params = np.array([norm_wp,var_wp1,var_wp2])
        else:
            raise ValueError(
                "Noise type not supported, should be either Gaussian or "
                "Non-Gaussian!"
            )
        

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator with the particles.

        Returns:
            particles: np.ndarray, dim: (num_states, num_particles)
                The particles corresponding to the initial state estimate. The
                order of states is given by x = [p_x, p_y, psi, tau, l].
        """
        R = self.constant.start_radius_bound
        psi_bar = self.constant.start_heading_bound
        px_py_samples = uniform(low=-R, high=R, size=(2,self.num_particles))
        psi_samples = uniform(low=-psi_bar, high=psi_bar, size=(1,self.num_particles))
        tau_samples = uniform(low=0, high=self.constant.start_velocity_bound, size=(1,self.num_particles))
        l_samples = np.linspace(self.constant.l_lb, self.constant.l_ub, self.num_particles).reshape((1,self.num_particles))
        particles = np.concatenate((px_py_samples, psi_samples,tau_samples,l_samples), axis = 0)
        return particles

    def estimate(
            self,
            particles: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the state of the vehicle.

        Args:
            particles : np.ndarray, dim: (num_states, num_particles)
                The posteriors of the particles of the previous time step k-1.
                The order of states is given by x = [p_x, p_y, psi, tau, l].
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [z_px, z_py, z_psi, z_tau].

        Returns:
            posteriors : np.ndarray, dim: (num_states, num_particles)
                The posterior particles at time step k. The order of states is
                given by x = [p_x, p_y, psi, tau, l].
        """
        

        (dim,n_particles) = particles.shape

        # PRIOR UPDATE
        process_noise = (uniform(low=-self.process_noise_bound, high=self.process_noise_bound, size=(n_particles,2))).T
        particles_next = process_model_PF(Ts=self.constant.Ts,xm_prev=particles,input=inputs, process_noise=process_noise)

        # MEASUREMENT UPDATE
        # GPS check
        no_GPS = False
        nan_end = 0
        if not np.all(np.logical_not(np.isnan(measurement))):
            no_GPS = True
            nan_end = 2
    
        # w is (dim x n_particles)
        w = (np.tile(measurement[nan_end:], (n_particles,1))).T-particles_next[nan_end:4,:]

        if no_GPS:
            pdf_z_x = pdf_w_evaluate_noGPS(x=w, psi_bound=self.psi_bound, tau_bound=self.tau_bound)

        else:
            pdf_z_x = pdf_w_evaluate_GPS(x=w, wp_params = self.wp_params, psi_bound=self.psi_bound, tau_bound=self.tau_bound)


        try:
            beta = pdf_z_x/np.sum(pdf_z_x)            

            posteriors_idx = np.random.choice(range(n_particles),size=n_particles,p=beta)
            posteriors = np.take(particles_next,posteriors_idx,axis=1)

        except:
            
            # Sample between particles and measurements
            r = np.random.rand(n_particles)
            particles_next[nan_end:4,:] = r*np.transpose(np.tile(measurement[nan_end:], (n_particles,1)))+(1-r)*particles_next[nan_end:4,:]
            particles_next[4,:] = uniform(low=self.constant.l_lb, high=self.constant.l_ub, size=(1,self.num_particles))
            posteriors=particles_next
        
        
        # Roughening
        K = 0.09
        E = np.ptp(posteriors, axis=1)
    
        roughening_sigma = np.maximum(K*E*(n_particles**(-1/dim)),0.001*np.ones(dim))
        roughening = np.random.normal(loc=0, scale=1, size=(dim, n_particles))
        roughening = roughening * roughening_sigma[:, np.newaxis]


        posteriors_rough = posteriors+roughening 
        
        return posteriors_rough




## Auxility Functions for EKF

def process_model_EKF(const: EstimatorConstant, xm_prev: np.ndarray, input: np.ndarray)->np.ndarray:
    
    Ts = const.Ts

    psi_prev = xm_prev[2]
    tau_prev = xm_prev[3]
    l_prev = xm_prev[4]

    b_prev = np.arctan(0.5*np.tan(input[0]))

    
    px = xm_prev[0] + tau_prev*np.cos(psi_prev + b_prev)*Ts
    py = xm_prev[1] + tau_prev*np.sin(psi_prev + b_prev)*Ts
    psi = psi_prev + (tau_prev/l_prev * np.sin(b_prev))*Ts
    tau = tau_prev + input[1]*Ts
    l = l_prev

    return [px,py,psi,tau,l]

def onlineMatrix_A(Ts:float, psi:float, tau:float, beta: float, l:float)->np.ndarray:
    A = np.eye(5,5)
    A[0,2] = -tau*np.sin(psi+beta)*Ts
    A[0,3] = np.cos(psi+beta)*Ts
    A[1,2] = tau*np.cos(psi+beta)*Ts
    A[1,3] = np.sin(psi+beta)*Ts
    A[2,3] = Ts*np.sin(beta)/l
    A[2,4] = -Ts*tau*np.sin(beta)/(l**2)
    return A

def offlineMatrix_L(Ts:float):
    L = np.zeros([5,2])
    L[2,0] = Ts
    L[3,1] = Ts
    return L





## Auxility Functions for PF

def process_model_PF(Ts: float, xm_prev: np.ndarray, input: np.ndarray, process_noise:np.ndarray)->np.ndarray:

    
    (dim,n_particles) = xm_prev.shape
   
    #px_prev = xm_prev[0,:]
    #py_prev = xm_prev[1,:]
    psi_prev = xm_prev[2,:]
    tau_prev = xm_prev[3,:]
    l_prev = xm_prev[4,:]

    udelta_prev = input[0]
    uc_prev = input[1]

    b_prev = np.arctan(0.5*np.tan(udelta_prev))

    v_b = np.array(process_noise[0,:])
    v_uc = np.array(process_noise[1,:])

    psi_beta = psi_prev + b_prev*np.ones((n_particles,))
    
    px = xm_prev[0,:] + tau_prev*np.cos(psi_beta)*Ts
    py = xm_prev[1,:] + tau_prev*np.sin(psi_beta)*Ts
    psi = psi_prev + (tau_prev/l_prev * np.sin(b_prev) + v_b)*Ts
    tau = tau_prev + (uc_prev + v_uc)*Ts
    #l = l_prev

    return np.array([px,py,psi,tau,l_prev])

def pdf_w_evaluate_noGPS(x:np.ndarray, psi_bound:float=None, tau_bound:float=None):
    
    pdf_psi = 1/(2*psi_bound) * np.where(  np.maximum ( psi_bound-abs(x[0,:]) , 0),1,0)
    pdf_tau = 1/(2*tau_bound) * np.where(  np.maximum ( tau_bound-abs(x[1,:]) , 0),1,0)
        
    result = pdf_psi*pdf_tau
    return result
    
def pdf_w_evaluate_GPS(x:np.ndarray, wp_params:np.ndarray, psi_bound:float=None, tau_bound:float=None):
    
    pdf_psi = 1/(2*psi_bound) * np.where(  np.maximum ( psi_bound-abs(x[2,:]) , 0),1,0)
    pdf_tau = 1/(2*tau_bound) * np.where(  np.maximum ( tau_bound-abs(x[3,:]) , 0),1,0)
    
    result = pdf_wp_quick(x[0,:],wp_params)*pdf_wp_quick(x[1,:],wp_params)*pdf_psi*pdf_tau
    return result

def pdf_wp_quick(x:np.ndarray, wp_params:np.ndarray)->np.ndarray:

    var_wp1 = wp_params[1]
    var_wp2 = wp_params[2]

    exp1 = np.exp ( var_wp1*( x - var_wp2 )**2  )
    exp2 = np.exp ( var_wp1*( x + var_wp2 )**2 )

    p = wp_params[0]*(exp1+exp2)

    return p