clear all;
addpath('E:/Work/os_lnd/source/matlab/lib')

system = 'two_spins';
type = 'classification';
num_epoches = 200;
suffix = 'ampl(0.5000_2.0000_50)_freq(0.0500_0.2000_50)_phase(0.0000_0.0000_0)';
models = {
    'resnet_scratch_prop_regular_class';
    };

x = linspace(1, num_epoches, num_epoches);

figures_path = sprintf('E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/%s/%s/figures/loss', system, type);
mkdir(figures_path);

fig = figure;

for model_id = 1:size(models, 1)
    model = string(models{model_id});
    model_split = split(model,'_');
    model = strcat(model, '_', suffix);
    path = sprintf('E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/%s/%s/%s', system, type, model);
    
    fn_txt = sprintf('%s/train_loss_%d.txt', path, num_epoches);
    train_loss = importdata(fn_txt);
    
    fn_txt = sprintf('%s/val_loss_%d.txt', path, num_epoches);
    val_loss = importdata(fn_txt);
    
    yyaxis left
    h = plot(x, train_loss, 'LineStyle', ':' , 'LineWidth', 2);
    hold all;
    legend(h, sprintf('%s train', model_split(1)));
    color = get(h, 'Color');
    h = plot(x, val_loss, 'LineStyle', '-' , 'LineWidth', 3, 'Color', color);
    legend(h, sprintf('%s val', model_split(1)));
    
    fn_txt = sprintf('%s/train_acc_%d.txt', path, num_epoches);
    train_loss = importdata(fn_txt);
    
    fn_txt = sprintf('%s/val_acc_%d.txt', path, num_epoches);
    val_loss = importdata(fn_txt);
    
    ylabel('CrossEntropyLoss', 'Interpreter', 'latex');
    
    yyaxis right
    h = plot(x, train_loss, 'LineStyle', ':' , 'LineWidth', 2);
    hold all;
    legend(h, sprintf('%s train', model_split(1)));
    color = get(h, 'Color');
    h = plot(x, val_loss, 'LineStyle', '-' , 'LineWidth', 3, 'Color', color);
    legend(h, sprintf('%s val', model_split(1)));
    
    ylabel('Accuracy', 'Interpreter', 'latex');
end

set(gca, 'FontSize', 30);
xlabel('epoch', 'Interpreter', 'latex');

fn_fig = sprintf('%s/loss_%d_%s', figures_path, num_epoches, suffix);
oqs_save_fig(fig, fn_fig);