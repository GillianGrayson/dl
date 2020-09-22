clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

path = 'E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/two_spins';

model = 'resnet_log_ampl(0.1000_0.1000_100)_freq(0.1000_0.1000_100)_phase(0.0000_0.0000_0)';
epochs = 100;

figures_path = sprintf('%s/%s/figures', path, model);
mkdir(figures_path);

gamma_1 = 0.05;
gamma_2 = 0.05;

ampl_begin = 0.5;
ampl_shift = 0.5;
ampl_num = 10;
ampl_chunks = 20;
ampl_stride = ampl_shift * ampl_num;

freq_begin = 0.05;
freq_shift = 0.05;
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

fn_txt = sprintf('%s/%s/norms_predicted_%d_%s.txt', path, model, epochs, suffix);
norms_predicted_dl = importdata(fn_txt);

fn_txt = sprintf('%s/%s/loss_%d_%s.txt', path, model, epochs, suffix);
loss_dl = importdata(fn_txt);

norms_original = zeros(ampl_num_global, freq_num_global);
norms_predicted = zeros(ampl_num_global, freq_num_global);
loss = zeros(ampl_num_global, freq_num_global);

ampls = linspace(ampl_begin, ampl_begin + (ampl_num_global - 1) * ampl_shift, ampl_num_global)';
freqs = linspace(freq_begin, freq_begin + (freq_num_global - 1) * freq_shift, freq_num_global)';

for ampl_id = 1:ampl_num_global
    for freq_id = 1:freq_num_global
        
        index = (ampl_id - 1) * freq_num_global + freq_id;
  
        norms_original(ampl_id, freq_id) = norm_dl_1(index);
        norms_predicted(ampl_id, freq_id) = norms_predicted_dl(index);
        loss(ampl_id, freq_id) = loss_dl(index);
        
    end
end

add_eps = min(norms_original(norms_original>0));
norms_original = log10(norms_original + add_eps);

fig = figure;
imagesc(ampls, freqs, norms_original');
set(gca, 'FontSize', 30);
xlabel('$A$', 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel('$\omega$', 'Interpreter', 'latex');
colormap hot;
h = colorbar;
set(gca, 'FontSize', 30);
title(h, '$\log_{10}(\mu_{REAL})$', 'FontSize', 33, 'interpreter','latex');
set(gca,'YDir','normal');
hold all;
fn_fig = sprintf('%s/norms_original_%d_%s', figures_path, epochs, suffix);
oqs_save_fig(fig, fn_fig);

fig = figure;
imagesc(ampls, freqs, norms_predicted');
set(gca, 'FontSize', 30);
xlabel('$A$', 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel('$\omega$', 'Interpreter', 'latex');
colormap hot;
h = colorbar;
set(gca, 'FontSize', 30);
title(h, '$\log_{10}(\mu_{PRED})$', 'FontSize', 33, 'interpreter','latex');
set(gca,'YDir','normal');
hold all;
fn_fig = sprintf('%s/norms_predicted_%d_%s', figures_path, epochs, suffix);
oqs_save_fig(fig, fn_fig);

fig = figure;
imagesc(ampls, freqs, loss');
set(gca, 'FontSize', 30);
xlabel('$A$', 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel('$\omega$', 'Interpreter', 'latex');
colormap hot;
h = colorbar;
set(gca, 'FontSize', 30);
title(h, '$loss$', 'FontSize', 33, 'interpreter','latex');
set(gca,'YDir','normal');
hold all;
fn_fig = sprintf('%s/loss_%d_%s', figures_path, epochs, suffix);
oqs_save_fig(fig, fn_fig);
