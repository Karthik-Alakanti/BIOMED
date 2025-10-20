import cupy as cp
import numpy as np

def VMD_gpu(signal, alpha, tau, K, DC, init, tol, N_iter=500):
    """
    GPU-accelerated Variational Mode Decomposition (CuPy version) - CORRECTED
    Returns: modes (K, N), spectra (K, N//2+1), central frequencies (K)
    """

    # Convert to GPU
    f = cp.asarray(signal)
    N = f.size

    # Frequency domain
    freqs = cp.fft.fftfreq(N)

    # Fourier transform of the signal
    f_hat = cp.fft.fft(f)
    
    # Initialization of modes and central frequencies
    u_hat = cp.zeros((K, N), dtype=cp.complex64)
    omega = cp.zeros((K, N_iter)) 
    
    if init == 1:
        initial_omegas = cp.linspace(0, 0.5, K)
        omega[:, 0] = initial_omegas
    elif init == 2:
        initial_omegas = cp.sort(cp.random.rand(K)) * 0.5
        omega[:, 0] = initial_omegas
    else:
        omega[:, 0] = cp.zeros(K)

    # Initialize Lagrange multiplier
    lambda_hat = cp.zeros(N, dtype=cp.complex64)
    
    # Store the sum of modes from the previous iteration for convergence check
    u_hat_old_sum = cp.zeros(N, dtype=cp.complex64)

    # Main optimization loop
    for n in range(1, N_iter):
        # Update each mode
        for k in range(K):
            # Sum of all modes except the current one
            u_hat_sum = cp.sum(u_hat, axis=0) - u_hat[k, :]
            
            # Update the mode in the frequency domain
            numerator = f_hat - u_hat_sum + lambda_hat / 2.
            denominator = 1. + 2. * alpha * (freqs - omega[k, n-1])**2
            u_hat[k, :] = numerator / denominator

            # --- BUG FIX: Correctly calculate the central frequency ---
            # Use the first half of the arrays, which correspond to positive frequencies
            freqs_pos = freqs[:N//2]
            u_hat_pos = u_hat[k, :N//2]
            
            # Calculate the power spectrum for the positive frequencies
            power_spectrum = cp.abs(u_hat_pos)**2
            
            # Update the central frequency as the weighted average of the power spectrum
            numerator_omega = cp.sum(freqs_pos * power_spectrum)
            denominator_omega = cp.sum(power_spectrum)
            
            # Avoid division by zero if a mode is empty
            if denominator_omega > 1e-10:
                omega[k, n] = numerator_omega / denominator_omega
            else:
                omega[k, n] = omega[k, n-1] # Keep the previous frequency
            
        # Update the Lagrange multiplier (dual ascent)
        u_hat_sum_new = cp.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * (u_hat_sum_new - f_hat)
        
        # Check for convergence
        diff = cp.sum(cp.abs(u_hat_sum_new - u_hat_old_sum)**2) / cp.sum(cp.abs(u_hat_old_sum)**2)
        u_hat_old_sum = u_hat_sum_new

        if diff < tol:
            break

    # Final central frequencies are the last calculated ones
    final_omegas = omega[:, n-1]
    
    # Inverse FFT to get time-domain modes
    u = cp.real(cp.fft.ifft(u_hat))

    # Bring back to CPU for compatibility with other libraries
    return cp.asnumpy(u), cp.asnumpy(u_hat), cp.asnumpy(final_omegas)

