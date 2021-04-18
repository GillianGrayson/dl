
x_tv_d = linspace(0.5, 98.5, 50);
y_tv_d = linspace(0.05, 9.85, 50);

N = 50;
x_tv_f = zeros(N * N, 1);
y_tv_f = zeros(N * N, 1);
for i = 1:N
    for j = 1:N
        index = (i - 1) * N + j; 
        x_tv_f(index) = x_tv_d(i);
        y_tv_f(index) = y_tv_d(j);
    end
end

x_t_d = linspace(0.2, 100, 500);
y_t_d = linspace(0.02, 10, 500);
N = 500;
x_t_f = zeros(N * N, 1);
y_t_f = zeros(N * N, 1);
for i = 1:N
    for j = 1:N
        index = (i - 1) * N + j; 
        x_t_f(index) = x_t_d(i);
        y_t_f(index) = y_t_d(j);
    end
end

hold all;


h = plot(x_tv_f, y_tv_f, 'o');
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
hold all;

h = plot(x_t_f, y_t_f, 'o');
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
hold all;

propertyeditor on;
grid on;
box on;

ololo = 1;

