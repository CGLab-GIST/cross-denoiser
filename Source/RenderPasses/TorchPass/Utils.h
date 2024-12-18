#pragma once

void print(const char *format, ...)
{
    va_list argptr;
    va_start(argptr, format);
    fprintf(stdout, "[TorchPass] ");
    vfprintf(stdout, format, argptr);
    va_end(argptr);
}

template <typename... Args>
void printOnce(const char *format, Args... args)
{
    static std::set<std::string> keys;
    auto res = keys.insert(format);
    if (!res.second)
        return;

    print(format, args...);
}

void printShape(const Tensor &tensor, const std::string &name, bool once = true)
{
    if (once)
    {
        static std::set<std::string> names;
        auto res = names.insert(name);
        if (!res.second)
            return;
    }

    printf("%s: (", name.c_str());

    for (size_t i = 0; i < tensor.sizes().size(); ++i)
    {
        printf("%lld", tensor.sizes()[i]);
        if (i < tensor.sizes().size() - 1)
            printf(" ");
    }
    printf(")\n");
}

Tensor relative_mse(Tensor y_true, Tensor y_pred)
{
    printShape(y_true, "y_true");
    printShape(y_pred, "y_pred");
    y_pred = torch::clamp_(y_pred, 0);

    auto true_mean = torch::mean(y_true, 0);
    printShape(true_mean, "true_mean");
    auto true_mean_squared = true_mean * true_mean;
    printShape(true_mean_squared, "true_mean_squared");
    auto diff_squared = torch::square(y_true - y_pred);
    printShape(diff_squared, "diff_squared");
    auto diff_squared_mean = torch::mean(diff_squared, 0);
    printShape(diff_squared_mean, "diff_squared_mean");
    auto res = diff_squared_mean / (true_mean_squared + 1e-2);
    printShape(res, "res");
    return res;
}
