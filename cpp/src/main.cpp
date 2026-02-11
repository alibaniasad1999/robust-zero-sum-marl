// main.cpp — CLI entry point for Robust Zero-Sum MARL training
// Replaces Python's src/train.py

#include <rzsm/agents/ddpg_agent.hpp>
#include <rzsm/agents/adversarial_ddpg_agent.hpp>
#include <rzsm/common.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --env ENV          Environment: Ant | Humanoid       (default: Ant)\n"
              << "  --mode MODE        Training mode: nominal | adversarial (default: nominal)\n"
              << "  --device DEV       Device: auto | cpu | cuda         (default: auto)\n"
              << "  --epochs N         Number of training epochs         (default: 100)\n"
              << "  --steps-per-epoch N Steps per epoch                  (default: 4000)\n"
              << "  --seed N           Random seed                       (default: 0)\n"
              << "  --hidden H1,H2     Hidden layer sizes                (default: 256,256)\n"
              << "  --pi-lr LR         Policy learning rate              (default: 1e-3)\n"
              << "  --q-lr LR          Q-function learning rate          (default: 1e-3)\n"
              << "  --batch-size N     Batch size                        (default: 1024)\n"
              << "  --gamma G          Discount factor                   (default: 0.99)\n"
              << "  --polyak TAU       Polyak averaging coefficient      (default: 0.995)\n"
              << "  --act-noise STD    Exploration noise std             (default: 0.1)\n"
              << "  --start-steps N    Random steps before policy        (default: 5000)\n"
              << "  --update-after N   Steps before updates begin        (default: 10000)\n"
              << "  --update-every N   Steps between update rounds       (default: 50)\n"
              << "  --n-updates N      Gradient steps per update round   (default: 500)\n"
              << "  --save-freq N      Save checkpoint every N epochs    (default: 10)\n"
              << "  --log-dir DIR      Log directory                     (default: auto)\n"
              << "\n  Adversarial-specific:\n"
              << "  --dist-ratio R     Disturbance ratio                 (default: 0.1)\n"
              << "  --dist-prob P      Disturbance probability           (default: 0.5)\n"
              << "  --pi-opt-path PATH Pre-trained nominal policy path   (default: \"\")\n"
              << "  --no-transformer   Disable transformer detector\n"
              << "  --det-seq-len N    Transformer sequence length       (default: 20)\n"
              << "  --det-d-model N    Transformer model dimension       (default: 128)\n"
              << "  --det-nhead N      Transformer attention heads       (default: 4)\n"
              << "  --det-layers N     Transformer encoder layers        (default: 3)\n"
              << "  --det-lr LR        Transformer learning rate         (default: 1e-4)\n"
              << "  --det-interval N   Transformer train interval        (default: 500)\n"
              << "\n  --help             Show this help message\n";
}

static std::vector<int64_t> parse_hidden(const std::string& s) {
    std::vector<int64_t> sizes;
    std::string token;
    for (char c : s) {
        if (c == ',') {
            sizes.push_back(std::stoll(token));
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) sizes.push_back(std::stoll(token));
    return sizes;
}

int main(int argc, char* argv[]) {
    // Defaults
    std::string env_name = "Ant";
    std::string mode = "nominal";
    std::string device_str = "auto";
    std::string log_dir;
    std::string hidden_str = "256,256";

    // Shared defaults
    int epochs = 100;
    int steps_per_epoch = 4000;
    int seed = 0;
    float pi_lr = 1e-3f;
    float q_lr = 1e-3f;
    int batch_size = 1024;
    float gamma = 0.99f;
    float polyak = 0.995f;
    float act_noise = 0.1f;
    int start_steps = 5000;
    int update_after = 10000;
    int update_every = 50;
    int n_updates = 500;
    int save_freq = 10;

    // Adversarial defaults
    float dist_ratio = 0.1f;
    float dist_prob = 0.5f;
    std::string pi_opt_path;
    bool use_transformer = true;
    int det_seq_len = 20;
    int det_d_model = 128;
    int det_nhead = 4;
    int det_layers = 3;
    float det_lr = 1e-4f;
    int det_interval = 500;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--env")             env_name = next();
        else if (arg == "--mode")            mode = next();
        else if (arg == "--device")          device_str = next();
        else if (arg == "--epochs")          epochs = std::stoi(next());
        else if (arg == "--steps-per-epoch") steps_per_epoch = std::stoi(next());
        else if (arg == "--seed")            seed = std::stoi(next());
        else if (arg == "--hidden")          hidden_str = next();
        else if (arg == "--pi-lr")           pi_lr = std::stof(next());
        else if (arg == "--q-lr")            q_lr = std::stof(next());
        else if (arg == "--batch-size")      batch_size = std::stoi(next());
        else if (arg == "--gamma")           gamma = std::stof(next());
        else if (arg == "--polyak")          polyak = std::stof(next());
        else if (arg == "--act-noise")       act_noise = std::stof(next());
        else if (arg == "--start-steps")     start_steps = std::stoi(next());
        else if (arg == "--update-after")    update_after = std::stoi(next());
        else if (arg == "--update-every")    update_every = std::stoi(next());
        else if (arg == "--n-updates")       n_updates = std::stoi(next());
        else if (arg == "--save-freq")       save_freq = std::stoi(next());
        else if (arg == "--log-dir")         log_dir = next();
        else if (arg == "--dist-ratio")      dist_ratio = std::stof(next());
        else if (arg == "--dist-prob")       dist_prob = std::stof(next());
        else if (arg == "--pi-opt-path")     pi_opt_path = next();
        else if (arg == "--no-transformer")  use_transformer = false;
        else if (arg == "--det-seq-len")     det_seq_len = std::stoi(next());
        else if (arg == "--det-d-model")     det_d_model = std::stoi(next());
        else if (arg == "--det-nhead")       det_nhead = std::stoi(next());
        else if (arg == "--det-layers")      det_layers = std::stoi(next());
        else if (arg == "--det-lr")          det_lr = std::stof(next());
        else if (arg == "--det-interval")    det_interval = std::stoi(next());
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    auto hidden = parse_hidden(hidden_str);

    // Validate env name
    if (env_name != "Ant" && env_name != "Humanoid") {
        std::cerr << "Error: --env must be Ant or Humanoid, got '" << env_name << "'\n";
        return 1;
    }

    // Validate mode
    if (mode != "nominal" && mode != "adversarial") {
        std::cerr << "Error: --mode must be nominal or adversarial, got '" << mode << "'\n";
        return 1;
    }

    auto device = rzsm::select_device(device_str);
    rzsm::set_seed(seed);

    std::cout << "╔══════════════════════════════════════════════════╗\n"
              << "║   Robust Zero-Sum MARL — C++ / LibTorch         ║\n"
              << "╚══════════════════════════════════════════════════╝\n"
              << "  Environment : " << env_name << "\n"
              << "  Mode        : " << mode << "\n"
              << "  Device      : " << device << "\n"
              << "  Epochs      : " << epochs << "\n"
              << "  Seed        : " << seed << "\n"
              << "  Hidden      : [";
    for (size_t i = 0; i < hidden.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << hidden[i];
    }
    std::cout << "]\n\n";

    // Environment factory (creates fresh env per call)
    auto env_fn = [&]() -> rzsm::MuJoCoEnv {
        return rzsm::MuJoCoEnv(env_name);
    };

    if (mode == "nominal") {
        rzsm::DDPGConfig cfg;
        cfg.hidden_sizes = hidden;
        cfg.seed = seed;
        cfg.steps_per_epoch = steps_per_epoch;
        cfg.epochs = epochs;
        cfg.pi_lr = pi_lr;
        cfg.q_lr = q_lr;
        cfg.batch_size = batch_size;
        cfg.gamma = gamma;
        cfg.polyak = polyak;
        cfg.act_noise = act_noise;
        cfg.start_steps = start_steps;
        cfg.update_after = update_after;
        cfg.update_every = update_every;
        cfg.n_updates = n_updates;
        cfg.save_freq = save_freq;
        cfg.device = device_str;
        cfg.log_dir = log_dir.empty() ? "logs/nominal_" + env_name : log_dir;

        std::cout << "Starting nominal DDPG training...\n\n";
        rzsm::DDPGAgent agent(env_fn, cfg);
        agent.train();

    } else {
        rzsm::AdversarialDDPGConfig cfg;
        cfg.hidden_sizes = hidden;
        cfg.seed = seed;
        cfg.steps_per_epoch = steps_per_epoch;
        cfg.epochs = epochs;
        cfg.pi_lr = pi_lr;
        cfg.q_lr = q_lr;
        cfg.batch_size = batch_size;
        cfg.gamma = gamma;
        cfg.polyak = polyak;
        cfg.act_noise = act_noise;
        cfg.start_steps = start_steps;
        cfg.update_after = update_after;
        cfg.update_every = update_every;
        cfg.n_updates = n_updates;
        cfg.save_freq = save_freq;
        cfg.device = device_str;
        cfg.log_dir = log_dir.empty() ? "logs/adversarial_" + env_name : log_dir;
        cfg.disturbance_ratio = dist_ratio;
        cfg.disturbance_probability = dist_prob;
        cfg.pi_opt_path = pi_opt_path;
        cfg.use_transformer = use_transformer;
        cfg.transformer_seq_len = det_seq_len;
        cfg.transformer_d_model = det_d_model;
        cfg.transformer_nhead = det_nhead;
        cfg.transformer_layers = det_layers;
        cfg.transformer_lr = det_lr;
        cfg.transformer_train_interval = det_interval;

        std::cout << "Starting adversarial DDPG training...\n"
                  << "  Disturbance ratio: " << dist_ratio << "\n"
                  << "  Transformer:       " << (use_transformer ? "enabled" : "disabled") << "\n";
        if (!pi_opt_path.empty())
            std::cout << "  Pre-trained pi_opt: " << pi_opt_path << "\n";
        std::cout << "\n";

        rzsm::AdversarialDDPGAgent agent(env_fn, cfg);
        agent.train();
    }

    std::cout << "\nTraining complete.\n";
    return 0;
}
