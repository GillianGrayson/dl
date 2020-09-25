clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

system = 'os';
N = 2;
N2 = N * N;
N4 = N2 * N2;

from = 1e-16;
to = 1e-4;

path = sprintf('E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/%s', system);

figures_path = sprintf('%s/figures', path);
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

fn_txt = sprintf('%s/props_dl_%s.txt', path, suffix);
props_dl = importdata(fn_txt);

pdf2d.x_num_bins = 500;
pdf2d.y_num_bins = 500;
pdf2d.x_label = '$Re(\lambda)$';
pdf2d.y_label = '$Im(\lambda)$';

pdf2d.x_bin_s = -1.05;
pdf2d.x_bin_f = 1.05;
pdf2d.y_bin_s = -1.05;
pdf2d.y_bin_f = 1.05;

pdf2d = oqs_pdf_2d_setup(pdf2d);

for ampl_id = 1:ampl_num_global
    for freq_id = 1:freq_num_global
        
        index = (ampl_id - 1) * freq_num_global + freq_id;
        
        norm = norm_dl_1(index);
        
        if (norm >= from) && (norm < to)
        
            prop_vec = props_dl((index - 1) * N4 + 1: index * N4, 1) + 1i * props_dl((index - 1) * N4 + 1: index * N4, 2);
            prop_mtx = zeros(N2);
            for row_id = 1:N2
                for col_id = 1:N2
                    prop_mtx(row_id, col_id) = prop_vec((row_id - 1) * N2 + col_id);
                end
            end

            evals = eig(prop_mtx);
            [max_val, max_id] = max(real(evals));
            if abs(max_val - 1.0) > 1e-13
                fprintf('ampl_id: %d \t freq_id: %d\n', ampl_id, freq_id);
            end
            evals(max_id) = []; 

            data2d = horzcat(real(evals), imag(evals));
            pdf2d = oqs_pdf_2d_update(pdf2d, data2d);

        end
    end
end

pdf2d = oqs_pdf_2d_release(pdf2d);

add_eps = min(min(pdf2d.pdf(pdf2d.pdf>0)));

fig = figure;
imagesc(pdf2d.x_bin_centers, pdf2d.y_bin_centers, log10(pdf2d.pdf' + add_eps));
set(gca, 'FontSize', 30);
xlabel(pdf2d.x_label, 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel(pdf2d.y_label, 'Interpreter', 'latex');
colormap hot;
h = colorbar;
set(gca, 'FontSize', 30);
title(h, '$PDF$', 'FontSize', 33, 'interpreter','latex');
set(gca,'YDir','normal');
hold all;

fn_fig = sprintf('%s/props_evals_from(%0.4f)_to(%0.4f)_%s', figures_path, log10(from), log10(to), suffix);
oqs_save_fig(fig, fn_fig)