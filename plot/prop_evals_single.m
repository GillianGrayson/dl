clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

system = 'two_spins';
N = 4;
N2 = N * N;
N4 = N2 * N2;

from = 0;
to = 1e-20;

path = sprintf('E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/%s', system);

figures_path = sprintf('%s/figures/props_evals/points', path);
mkdir(figures_path);

ampl_begin = 0.2;
ampl_shift = 0.2;
ampl_num = 10;
ampl_chunks = 50;
ampl_stride = ampl_shift * ampl_num;

freq_begin = 0.02;
freq_shift = 0.02;
freq_num = 10;
freq_chunks = 50;
freq_stride = freq_shift * freq_num;
ph = 0;

ampl_targ = 97;
ampl_id_targ = round((ampl_targ - ampl_begin) / ampl_shift) + 1;
freq_targ = 1.14;
freq_id_targ = round((freq_targ - freq_begin) / freq_shift) + 1;

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

index = (ampl_id_targ - 1) * freq_num_global + freq_id_targ;

norm = norm_dl_1(index);

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

th = 0:pi/100:2*pi;
xunit = cos(th);
yunit = sin(th);

fig = figure;
plot(real(evals), imag(evals), 'o', 'LineWidth', 1, 'MarkerFaceColor', [1 1 1]);
set(gca, 'FontSize', 30);
xlabel('$Re(\lambda)$', 'Interpreter', 'latex');
set(gca, 'FontSize', 30);
ylabel('$Im(\lambda)$', 'Interpreter', 'latex');
xlim([-1.05 1.05])
ylim([-1.05 1.05])
hold all;
h = plot(xunit, yunit, 'LineWidth', 1);
hold off

suffix = sprintf('ampl(%0.4f)_freq(%0.4f)_phase(%0.4f)', ...
    ampl_targ, ...
    freq_targ, ...
    ph);

fn_fig = sprintf('%s/props_evals_%s', figures_path, suffix);
oqs_save_fig(fig, fn_fig)
