clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

system = 'os';
N = 2;
N2 = N * N;
N4 = N2 * N2;

is_smooth = 0;
is_normalize = 1;

path = sprintf('E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/%s', system);

figures_path = sprintf('%s/figures/props_evals', path);
mkdir(figures_path);

ampl_begin = 0.02;
ampl_shift = 0.02;
ampl_num = 10;
ampl_chunks = 20;
ampl_stride = ampl_shift * ampl_num;

freq_begin = 0.035;
freq_shift = 0.035;
freq_num = 10;
freq_chunks = 20;
freq_stride = freq_shift * freq_num;
ph = 0;

ampl_num_global = ampl_num * ampl_chunks;
freq_num_global = freq_num * freq_chunks;

suffix = sprintf('ampl(%0.4f_%0.4f_%d)_freq(%0.4f_%0.4f_%d)_phase(%0.4f_%0.4f_%d)', ...
    ampl_begin, ...
    ampl_shift, ...
    ampl_num_global, ...
    freq_begin, ...
    freq_shift, ...
    freq_num_global, ...
    ph, ...
    0, ...
    0);

fn_txt = sprintf('%s/ampls_dl_%s.txt', path, suffix);
ampls_dl = importdata(fn_txt);

fn_txt = sprintf('%s/freqs_dl_%s.txt', path, suffix);
freqs_dl = importdata(fn_txt);

fn_txt = sprintf('%s/norm_dl_1_%s.txt', path, suffix);
norm_dl_1 = importdata(fn_txt);

mus_log = log10(norm_dl_1 + min(norm_dl_1(norm_dl_1>0)));

pdf_mu.x_num_bins = 200;
pdf_mu.x_label = '$\log_{10}\mu_{min}$';
pdf_mu.x_bin_s = min(mus_log);
pdf_mu.x_bin_f = max(mus_log);
pdf_mu = oqs_pdf_1d_setup(pdf_mu);
pdf_mu = oqs_pdf_1d_update(pdf_mu, mus_log);
pdf_mu = oqs_pdf_1d_release(pdf_mu);
fig = figure;
plot(pdf_mu.x_bin_centers, log10(pdf_mu.pdf + 1e-6), 'LineWidth', 2);
set(gca, 'FontSize', 30);
xlabel(pdf_mu.x_label, 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel('$\log_{10}PDF$', 'Interpreter', 'latex');
hold all;
fn_fig = sprintf('%s/mus_pdf_%s', figures_path, suffix);
oqs_save_fig(fig, fn_fig);

num_mu = 100;
mu_s = min(mus_log);
mu_f = max(mus_log);
mu_shift = (mu_f - mu_s) / num_mu;
mu_centers = linspace(...
    mu_s + 0.5 * mu_shift, ...
    mu_f - 0.5 * mu_shift, ...
    num_mu)';

fn_txt = sprintf('%s/props_dl_%s.txt', path, suffix);
props_dl = importdata(fn_txt);

ratio_pdf.xs = mu_centers;
ratio_pdf.x_num_points = num_mu;
ratio_pdf.y_num_bins = 100;
ratio_pdf.x_label = '$\log_{10}\mu_{min}$';
ratio_pdf.y_label = '$Re(\lambda)$';
ratio_pdf.y_bin_s = -1.05;
ratio_pdf.y_bin_f = 1.05;
ratio_pdf = oqs_pdf_2d_lead_by_x_setup(ratio_pdf);

for ampl_id = 1:ampl_num_global
    for freq_id = 1:freq_num_global
        
        index = (ampl_id - 1) * freq_num_global + freq_id;
        
        norm = mus_log(index);
        
        x_id = floor((norm - mu_s) / (mu_shift + 1e-10)) + 1;
       
        prop_vec = props_dl((index - 1) * N4 + 1: index * N4, 1) + 1i * props_dl((index - 1) * N4 + 1: index * N4, 2);
        prop_mtx = zeros(N2);
        for row_id = 1:N2
            for col_id = 1:N2
                prop_mtx(row_id, col_id) = prop_vec((row_id - 1) * N2 + col_id);
            end
        end
        
        evals = eig(prop_mtx);
        [max_val, max_id] = max(real(evals));
        if abs(max_val - 1.0) > 1e-11
            fprintf('ampl_id: %d \t freq_id: %d\n', ampl_id, freq_id);
        end
        evals(max_id) = [];
        
        ratio_pdf = oqs_pdf_2d_lead_by_x_update_slice(ratio_pdf, real(evals), x_id);
        
    end
end

ratio_pdf = oqs_pdf_2d_lead_by_x_release(ratio_pdf, is_smooth, is_normalize);
fig = oqs_pdf_2d_lead_by_x_plot(ratio_pdf);
fn_fig = sprintf('%s/real_evals_from_log_mu_%s', figures_path, suffix);
oqs_save_fig(fig, fn_fig)

