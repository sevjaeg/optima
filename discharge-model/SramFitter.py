import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from util import *

class SramFitter:
    """

    """

    def __init__(self, fit_time, nominal_vdd=NOMINAL_VDD, nominal_temp=NOMINAL_TEMP):
        self.fit_time   = fit_time
        self.nominal_vdd  = nominal_vdd
        self.nominal_temp = nominal_temp
        self.model_size_vdd = 11
        self.model_size_temp = 4
        self.model_size_mismatch = 7
        self.model_size_e_write = 4
        self.model_size_e_mul = 9
        self.clear_fit()

    def clear_fit(self):
        self.bl_fitted       = False
        self.temp_fitted     = False
        self.mismatch_fitted = False
        self.e_write_fitted  = False
        self.e_mul_fitted    = False
        self.bl_params       = np.zeros(self.model_size_vdd)
        self.temp_params     = np.zeros(self.model_size_temp)
        self.mismatch_params = np.zeros(self.model_size_mismatch)
        self.e_write_params  = np.zeros(self.model_size_e_write)
        self.e_mul_params    = np.zeros(self.model_size_e_mul)

    def any_fit(self):
        return self.bl_fitted or self.temp_fitted or self.mismatch_fitted or self.e_write_fitted or self.e_mul_fitted

    def filter_rows(self, t, vbl, vwl, vdd=None, temp=None, sigma=None):
        filter = np.argwhere(t <= self.fit_time)
        t   = t[filter].flatten()

        if isinstance(vwl, np.ndarray):
            vwl = vwl[filter].flatten()

        if isinstance(vbl, np.ndarray):
            vbl = vbl[filter].flatten()

        if isinstance(vdd, np.ndarray):
            vdd = vdd[filter].flatten()

        if isinstance(temp, np.ndarray):
            temp = temp[filter].flatten()

        if isinstance(sigma, np.ndarray):
            sigma = sigma[filter].flatten()

        return t,vbl,vwl,vdd,temp,sigma

    def bit_line_function(self, xyz : tuple, *params : list) -> None:
        t, vwl, vdd = xyz
        vwl = np.where(vwl > 0.3, vwl - 0.3, 0.0)
        vdd_diff = vdd - NOMINAL_VDD
        t = t*1e9  # convert to ns to keep coefficents in a reasonable range
        return vdd + (params[0] + params[1]*t + params[2]*t**2) * (params[3] + params[4]*vwl + params[5]*vwl**2 +params[6]*vwl**3 + params[7]*vwl**4)  * (params[8] + params[9]*vdd_diff + params[10]*vdd_diff**2)

    def temperature_error_function(self, xyz, *params):
        t, vwl, temp = xyz
        t = t*1e9  # convert to ns to keep coefficents in a reasonable range
        return t * temp * (params[0] + params[1]*vwl + params[2]*vwl**2 + params[3]*vwl**3)

    def mismatch_sigma_function(self, xy, *params):
        t, vwl = xy
        t = t*1e9  # convert to ns to keep coefficents in a reasonable range
        vwl = np.where(vwl > 0.3, vwl - 0.3, 0.0)
        return (params[0] + params[1]*t  + params[2]*t**3)  * ( params[3] + params[4]*vwl + params[5]*vwl**2 + params[6]*vwl**3)

    def write_energy_function(self, xyz, *params):
        data, vdd, temp = xyz
        energy = (params[0]+ params[1]*vdd + params[2]*(vdd**2)) * (1+params[3]*temp)
        return energy

    def discharge_energy_function(self, wxyz, *params):
        data, vdd, temp, vwl, discharge = wxyz
        discharge = vdd - discharge
        vwl = vwl - 0.3

        dc = discharge * (1 + data * (params[5] + (params[6]*(discharge-0.3))) * (params[7] + params[8]*vwl))
        energy = (params[0] + vdd) * (params[1]+params[2]*dc+params[3]*dc**2) * (1 + params[4]*temp)
        return energy

    def print_params(self):
        if not self.any_fit():
            print("No model fitted yet\n")
            return()

        print("\n------------ Fitting Results ------------")

        if self.bl_fitted:
            print("Bit line fit [V]:\n\tvbl = vdd +\n\t(%.5f + %.5f*t + %.5f*t**2) * (%.5f + %.5f*vwl + %.5f*vwl**2 + %.5f*vwl**3 + %.5f*vwl**4) *\n\t(%.5f + %.5f*vdd + %.5f*vdd**2)" \
                % (self.bl_params[0], self.bl_params[1], self.bl_params[2], self.bl_params[3], self.bl_params[4], self.bl_params[5], self.bl_params[6], self.bl_params[7], self.bl_params[8], self.bl_params[9], self.bl_params[10]))

        if self.temp_fitted:
            print("Temperature error fit [V]:\n\tdelta_vbl = t * delta_temp * (%.5f + %.5f*vwl + %.5f*vwl**2 + %.5f*vwl**3)" \
                   % (self.temp_params[0], self.temp_params[1], self.temp_params[2], self.temp_params[3]))

        if self.mismatch_fitted:
            print("Mismatch sigma fit [V]:\n\tsigma = (%.5f + %.5f*t + %.5f*vwl**3) * (%.5f + %.5f*vwl + %.5f*vwl**2 + %.5f*vwl**3)" % (self.mismatch_params[0], self.mismatch_params[1], self.mismatch_params[2], self.mismatch_params[3], self.mismatch_params[4], self.mismatch_params[5], self.mismatch_params[6]))

        if self.e_write_fitted:
            p = 1e15 * self.e_write_params
            print("Write energy fit [fJ]:\n\tp_wr = (%.5e + %.5e * vdd +  %.5e * vdd**2) * (1 + %.5e * temp)" % (p[0], p[1], p[2], p[3]))

        if self.e_mul_fitted:
            p = 1e15 * self.e_mul_params
            print("Discharge energy fit [fJ]:\n\tp_wr = (%.5e + vdd) * (1 + %.5e*temp) * (%.5e + %.5e * dc +  %.5e * dc**2)\n\twith dc = bl * (1 + data * (%.5e + (%.5e*(bl-0.3))) * (%.5e + %.5e*vwl))" % (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))


    def fit_bit_line(self, vbl, t, vwl, vdd, bounds=(-8, 8)):
        t, vbl, vwl, vdd, _, _ = self.filter_rows(t, vbl, vwl, vdd=vdd)

        self.bl_params, covariance = curve_fit(self.bit_line_function, (t, vwl, vdd), vbl, bounds=bounds, p0=np.ones(self.model_size_vdd))
        self.bl_fitted = True

    def fit_temperature(self, vbl, t, vwl, temp, bounds=(-5, 5)):
        t, vbl, vwl, _, temp, _ = self.filter_rows(t, vbl, vwl, temp=temp)

        idx_nominal = np.argwhere(temp == self.nominal_temp)
        vbl_diff = np.copy(vbl)
        for wl_voltage in np.unique(vwl):
            idx_wl = np.argwhere(vwl == wl_voltage)
            idx_nominal_temp = np.intersect1d(idx_nominal, idx_wl)
            for temperature in np.unique(temp):
                idx_temp = np.intersect1d(np.argwhere(temp == temperature), idx_wl)
                vbl_diff[idx_temp] = vbl[idx_temp] - vbl[idx_nominal_temp]
        temp_diff = temp - self.nominal_temp

        self.temp_params, covariance = curve_fit(self.temperature_error_function, (t, vwl, temp_diff), vbl_diff, bounds=bounds, p0=np.ones(self.model_size_temp))
        self.temp_fitted = True

    def fit_mismatch(self, t, vwl, sigma, bounds=(-5, 5)):
        t, _, vwl, _, _, sigma = self.filter_rows(t, None, vwl, sigma=sigma)
        self.mismatch_params, covariance = curve_fit(self.mismatch_sigma_function, (t, vwl), sigma, bounds=bounds, p0=np.ones(self.model_size_mismatch))
        self.mismatch_fitted = True

    def fit_write_energy(self, data, vdd, temp, energy):
        self.e_write_params, covariance = curve_fit(self.write_energy_function, (data, vdd, temp), energy, p0=np.ones(self.model_size_e_write))
        self.e_write_fitted = True

    def fit_discharge_energy(self, data, vdd, temp, vwl, energy, bounds=(-1e-8, 1e-8)):
        discharge = self.sample_discharge(1.6e-9, vwl, vdd, temp, mismatch=False)
        self.e_mul_params, covariance = curve_fit(self.discharge_energy_function, (data, vdd, temp, vwl, discharge), energy, p0=np.ones(self.model_size_e_mul), maxfev=250000)
        self.e_mul_fitted = True

    def sample_bit_line(self, t, vwl, vdd):
        return np.array(self.bit_line_function((t, vwl, vdd), *self.bl_params))

    def sample_temperature(self, t, vwl, temp):
        temp_diff = temp - self.nominal_temp
        return self.temperature_error_function((t, vwl, temp_diff), *self.temp_params)

    def sample_mismatch_sigma(self, t, vwl):
        return self.mismatch_sigma_function((t, vwl), *self.mismatch_params)

    def sample_write_energy(self, data, vdd, temp):
        return self.write_energy_function((data, vdd, temp), *self.e_write_params)

    def sample_discharge_energy(self, data, vdd, temp, vwl):
        discharge = self.sample_discharge(1.6e-9, vwl, vdd, temp, mismatch=False)
        return self.discharge_energy_function((data, vdd, temp, vwl, discharge), *self.e_mul_params)

    def sample_discharge(self, t, vwl, vdd, temp, mismatch=False, mismatch_sample=None):
        vbl_basic = self.sample_bit_line(t, vwl, vdd)
        vbl_temp = vbl_basic + self.sample_temperature(t, vwl, temp)
        if mismatch:
            sigma = self.sample_mismatch_sigma(t, vwl)
            if not mismatch_sample:  # sample new random value (as not given)
                mismatch_sample = np.random.default_rng().normal(loc=0.0, scale=1)
            vbl_mismatch = vbl_temp + mismatch_sample*sigma
            ret = vbl_mismatch
        else:
            ret = vbl_temp
        ret = np.clip(ret, 0.0, vdd)
        return ret

    def validate_model(self, t, vwl, vdd, vbl, temp, mismatch=False, plot=False,
                       plot_err_details=False, savepath=None, single_plot=False) -> None:
        # t, vbl, vwl, vdd, temp, _ = self.filter_rows(t, vbl, vwl, vdd, temp=temp)

        vbl_model = self.sample_discharge(t, vwl, vdd, temp, mismatch)
        error = vbl - vbl_model
        print("Error avg: %.4f mV, max: %.4f mV, rms: %.4f mV" % (1e3*np.sum(np.abs(error))/error.size, 1e3*np.max(np.abs(error)), 1e3*np.std(error)))
        ref = (vdd - vbl)
        ref = np.where(ref == 0.0, np.inf, ref)
        rel_error = error / ref
        rel_error = np.where(rel_error == np.nan, 0, rel_error)
        print("Relative error avg: %.4f, max: %.4f\n" % (np.sum(np.abs(rel_error))/rel_error.size, np.max(np.abs(rel_error))))
        if plot:
            plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
            if single_plot:
                marker='None'
                linea = 'dashed'
                lineb = 'dotted'
            else:
                marker = 'x'
                linea  = 'None'
                lineb  = 'None'
            plt.plot(t*1e9, vbl, color ='red', marker=marker, markersize=2.5, label ="Spice", linestyle=lineb)
            plt.plot(t*1e9, vbl_model, color ='blue', marker=marker, markersize=2.5, label ="Fitted",linestyle=linea)
            plt.legend()
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf V_{\overline{BL}}$ \bf [V]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Bit line voltage')
            plt.show()
        if plot_err_details:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(t*1e9, 1000*error, linestyle='None', marker='x',color ='red')
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf \Delta V_{\overline{BL}}$ \bf [mV]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + "-err.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$V_{\overline{BL}}$ modelling error')
            plt.show()

            hist, bins = np.histogram(1000*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"\bf Error magnitude [mV]")
            plt.ylabel(r"\bf Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$V_{\overline{BL}}$ modelling error')
            plt.show()

    def validate_temperature_model(self, t, vbl, vwl, temp, plot=False, plot_err_details=False, savepath=None) -> None:
        t, vbl, vwl, _ , temp, _ = self.filter_rows(t, vbl, vwl, temp=temp)

        idx_nominal = np.argwhere(temp == self.nominal_temp)
        vbl_diff = np.copy(vbl)
        for wl_voltage in np.unique(vwl):
            idx_wl = np.argwhere(vwl == wl_voltage)
            idx_nominal_temp = np.intersect1d(idx_nominal, idx_wl)
            for temperature in np.unique(temp):
                idx_temp = np.intersect1d(np.argwhere(temp == temperature), idx_wl)
                vbl_diff[idx_temp] = vbl_diff[idx_temp] - vbl[idx_nominal_temp]

        vbl_model = self.sample_temperature(t, vwl, temp)
        error = vbl_diff - vbl_model
        print("Error avg: %.4f mV, max: %.4f mV, rms: %.4f mV" % (1e3*np.sum(np.abs(error))/error.size, 1e3*np.max(np.abs(error)), 1e3*np.std(error)))
        ref = self.nominal_vdd - vbl
        ref = np.where(ref == 0.0, np.inf, ref)
        rel_error = error / ref
        rel_error = np.where(rel_error == np.nan, 0, rel_error)
        print("Relative error avg: %.4f, max: %.4f\n" % (np.sum(np.abs(rel_error))/rel_error.size, np.max(np.abs(rel_error))))
        if plot:
            plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
            marker = 'x'
            linea  = 'None'
            lineb  = 'None'
            plt.plot(t*1e9, 1000*vbl_diff, color ='red', marker=marker, markersize=2.5, label ="Spice", linestyle=lineb)
            plt.plot(t*1e9, 1000*vbl_model, color ='blue', marker=marker, markersize=2.5, label ="Fitted",linestyle=linea)
            plt.legend()
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf \Delta V_{\overline{BL}}$ \bf [mV]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Bit line voltage')
            plt.show()
        if plot_err_details:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(t*1e9, 1000*error, linestyle='None', marker='x',color ='red')
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf \Delta V_{\overline{BL}}$ \bf [mV]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + "-err.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$V_{\overline{BL}}$ modelling error')
            plt.show()

            hist, bins = np.histogram(1000*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"\bf Error magnitude [mV]")
            plt.ylabel("Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$V_{\overline{BL}}$ modelling error')
            plt.show()

    def validate_mismatch_model(self, t, sigma, vwl, plot=False, plot_err_details=False, savepath=None) -> None:
        t, _, vwl, _, _, sigma = self.filter_rows(t, None, vwl, sigma=sigma)

        sigma_model = self.sample_mismatch_sigma(t, vwl)
        error = sigma - sigma_model
        print("Error avg: %.4f mV, max: %.4f mV, rms: %.4f mV" % (1e3*np.sum(np.abs(error))/error.size, 1e3*np.max(np.abs(error)), 1e3*np.std(error)))
        if plot:
            plt.figure(figsize=(FIG_WIDTH_WIDE*CM, FIG_HEIGHT*CM))
            plt.plot(t*1e9, 1000*sigma, color ='red', marker='x', markersize=2.5, label ="Spice", linestyle='None')
            plt.plot(t*1e9, 1000*sigma_model, color ='blue', marker='x', markersize=2.5, label ="Fitted", linestyle='None')
            plt.legend()
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf \sigma$ \bf [mV]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Mismatch-induced bit line voltage standard deviation')
            plt.show()
        if plot_err_details:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(t*1e9, 1000*error, linestyle='None', marker='x',color ='red')
            plt.xlabel(r"$\bf t$ \bf [ns]")
            plt.ylabel(r"$\bf \Delta V_{\overline{BL}}$ $\sigma$ \bf [mV]")
            plt.grid()
            # plt.xlim(left=0.0, right=1.8)
            if savepath:
                plt.savefig(savepath + "-err.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$V_{\overline{BL}}$ $\sigma$ modelling error')
            plt.show()

            hist, bins = np.histogram(1000*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"\bf Error magnitude [mV]")
            plt.ylabel("Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$\sigma$ modelling error')
            plt.show()

    def validate_energy_model(self, data, vwl, vdd, temp, e_wr, e_disch, plot=False, plot_err_details=False, savepath=None) -> None:
        print("Write energy")
        e_wr_model = self.sample_write_energy(data, vdd, temp)
        error = e_wr - e_wr_model
        print("Error avg: %.4f fJ, max: %.4f fJ, rms: %.4f fJ" % (1e15*np.sum(np.abs(error))/error.size, 1e15*np.max(np.abs(error)), 1e15*np.std(error)))

        if plot:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(vdd, 1e15*e_wr, color ='red', marker='x', markersize=2.5, label ="Spice", linestyle='None')
            plt.plot(vdd, 1e15*e_wr_model, color ='blue', marker='x', markersize=2.5, label ="Fitted", linestyle='None')
            plt.legend()
            plt.xlabel(r"$\bf V_{DD}$ \bf [V]")
            plt.ylabel(r"$\bf E$ \bf [fJ]")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-write.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Write energy')
            plt.show()
        if plot_err_details:
            hist, bins = np.histogram(1e15*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"$\bf \Delta E$ \bf [fJ]")
            plt.ylabel(r"\bf Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-write-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$\sigma$ modelling error')
            plt.show()

        print("Discharge energy")
        e_dc_model = self.sample_discharge_energy(data, vdd, temp, vwl)
        error = e_disch - e_dc_model
        print("Error avg: %.4f fJ, max: %.4f fJ, rms: %.4f fJ" % (1e15*np.sum(np.abs(error))/error.size, 1e15*np.max(np.abs(error)), 1e15*np.std(error)))

        if plot:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(vdd[(data==0.0)], 1e15*e_disch[(data==0.0)], color ='red', marker='x', markersize=2.5, label ="Spice 0", linestyle='None')
            plt.plot(vdd[(data==0.0)], 1e15*e_dc_model[(data==0.0)], color ='blue', marker='x', markersize=2.5, label ="Fitted 0", linestyle='None')
            plt.plot(vdd[(data==1.0)], 1e15*e_disch[(data==1.0)], color ='orange', marker='o', markersize=2.5, label ="Spice 1", linestyle='None')
            plt.plot(vdd[(data==1.0)], 1e15*e_dc_model[(data==1.0)], color ='purple', marker='o', markersize=2.5, label ="Fitted 1", linestyle='None')
            plt.legend()
            plt.xlabel(r"$\bf V_{DD}$ \bf [V]")
            plt.ylabel(r"$\bf E$ \bf [fJ]")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-discharge.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Discharge energy')
            plt.show()
        if plot_err_details:
            hist, bins = np.histogram(1e15*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"$\bf \Delta E$ \bf [fJ]")
            plt.ylabel(r"\bf Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-discharge-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$\sigma$ modelling error')
            plt.show()
        if plot:
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.plot(vwl[(data==0.0)], 1e15*e_disch[(data==0.0)], color ='red', marker='x', markersize=2.5, label ="Spice 0", linestyle='None')
            plt.plot(vwl[(data==0.0)], 1e15*e_dc_model[(data==0.0)], color ='blue', marker='x', markersize=2.5, label ="Fitted 0", linestyle='None')
            plt.plot(vwl[(data==1.0)], 1e15*e_disch[(data==1.0)], color ='orange', marker='x', markersize=2.5, label ="Spice 1", linestyle='None')
            plt.plot(vwl[(data==1.0)], 1e15*e_dc_model[(data==1.0)], color ='purple', marker='x', markersize=2.5, label ="Fitted 1", linestyle='None')
            plt.legend()
            plt.xlabel(r"$\bf V_{WL}$ \bf [V]")
            plt.ylabel(r"$\bf E$ \bf [fJ]")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-discharge-vwl.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('Discharge energy')
            plt.show()
        if plot_err_details:
            hist, bins = np.histogram(1e15*error, 31)
            plt.figure(figsize=(FIG_WIDTH_SMALL*CM, FIG_HEIGHT*CM))
            plt.stairs(hist, edges=bins,color ='red')
            plt.xlabel(r"$\bf \Delta E$ \bf [fJ]")
            plt.ylabel(r"\bf Number of occurrences")
            plt.grid()
            if savepath:
                plt.savefig(savepath + "-discharge-vwl-err-hist.pdf", format='pdf', bbox_inches='tight', pad_inches=0)
            else:
                plt.title('$\sigma$ modelling error')
            plt.show()


    def export_params(self, path, name : str):
        if not self.any_fit():
            print("No model fitted yet\n")
            return()

        filename = os.path.join(path, name)

        with open(filename, 'w') as f:
            f.write("/*\n * Generated file. Do not modify.\n */\n\n")

            f.write("`define VDD_MODEL_SIZE %d\n" % self.model_size_vdd)
            f.write("`define TEMP_MODEL_SIZE %d\n" % self.model_size_temp)
            f.write("`define MISMATCH_MODEL_SIZE %d\n" % self.model_size_mismatch)
            f.write("`define WRITE_ENERGY_MODEL_SIZE %d\n" % self.model_size_e_write)
            f.write("`define DISCHARGE_ENERGY_MODEL_SIZE %d\n\n" % self.model_size_e_mul)

            for i in range(self.model_size_vdd):
                f.write("localparam real model_bl_vdd_%d = %.5e;\n" % (i, self.bl_params[i]))
            f.write("\n")

            for i in range(self.model_size_temp):
                f.write("localparam real model_bl_temp_%d = %.5e;\n" % (i, self.temp_params[i]))
            f.write("\n")

            for i in range(self.model_size_mismatch):
                f.write("localparam real model_bl_mismatch_%d = %.5e;\n" % (i, self.mismatch_params[i]))
            f.write("\n")

            for i in range(self.model_size_e_write):
                f.write("localparam real model_en_write_%d = %.5e;\n" % (i, self.e_write_params[i]))
            f.write("\n")

            for i in range(self.model_size_e_mul):
                f.write("localparam real model_en_discharge_%d = %.5e;\n" % (i, self.e_mul_params[i]))
            f.write("\n")

        print("\nParameters exported to %s\n" % filename)


if __name__ == "__main__":
    fitter = SramFitter(2e-9)
