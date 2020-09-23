clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

path = 'E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/ospm';

ampl_begin = 0.25;
ampl_shift = 0.25;
ampl_num = 10;
ampl_chunks = 20;
ampl_stride = ampl_shift * ampl_num;

freq_begin = 0.025;
freq_shift = 0.025;
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

fn_txt = sprintf('%s/norm_dl_1_%s.txt', path, suffix);
norm_dl_1 = importdata(fn_txt);

norms_original = zeros(ampl_num_global, freq_num_global);
norms_corrected = zeros(ampl_num_global, freq_num_global);

ampls = linspace(ampl_begin, ampl_begin + (ampl_num_global - 1) * ampl_shift, ampl_num_global)';
freqs = linspace(freq_begin, freq_begin + (freq_num_global - 1) * freq_shift, freq_num_global)';

norm_corrected_dl = zeros(ampl_num_global * freq_num_global, 1);

for ampl_id = 1:ampl_num_global
    for freq_id = 1:freq_num_global
        
        index = (ampl_id - 1) * freq_num_global + freq_id;
  
        norms_original(ampl_id, freq_id) = norm_dl_1(index);
      
        norms_corrected(ampl_id, freq_id) = norms_original(ampl_id, freq_id);
        if abs(norms_corrected(ampl_id, freq_id) - 1) < 1e-4
            
            neighborhood_ampls = [];
            if ampl_id > 1
                neighborhood_ampls = vertcat(neighborhood_ampls, ampl_id - 1);
            end
            neighborhood_ampls = vertcat(neighborhood_ampls, ampl_id);
            if ampl_id < ampl_num_global
                neighborhood_ampls = vertcat(neighborhood_ampls, ampl_id + 1);
            end
            
            neighborhood_freqs = [];
            if freq_id > 1
                neighborhood_freqs = vertcat(neighborhood_freqs, freq_id - 1);
            end
            neighborhood_freqs = vertcat(neighborhood_freqs, freq_id);
            if freq_id < freq_num_global
                neighborhood_freqs = vertcat(neighborhood_freqs, freq_id + 1);
            end
            
            count = 0;
            sum = 0.0;
            for a_n_id = 1:size(neighborhood_ampls, 1)
                for f_n_id = 1:size(neighborhood_freqs, 1)
                    if (neighborhood_ampls(a_n_id) ~= ampl_id) || (neighborhood_freqs(f_n_id) ~= freq_id)
                        sum = sum + norms_corrected(neighborhood_ampls(a_n_id), neighborhood_freqs(f_n_id));
                        count = count + 1;
                    end
                end
            end
            norms_corrected(ampl_id, freq_id) = sum / count;
            
        end
        
        global_index = (ampl_id - 1) * freq_num_global + freq_id;
        
        norm_corrected_dl(global_index) = norms_corrected(ampl_id, freq_id);
        
    end
end

norms_original = log10(norms_original);
norms_corrected = log10(norms_corrected);

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

fig = figure;
imagesc(ampls, freqs, norms_corrected');
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

fn_txt = sprintf('%s/norm_dl_1_corrected_%s.txt', path, suffix);
fid = fopen(fn_txt,'wt');
for x_id = 1:size(norm_corrected_dl, 1)
    fprintf(fid,'%0.16e\n', norm_corrected_dl(x_id));
end
fclose(fid);
