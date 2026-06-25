#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct rpp_live_slot_info {
    int32_t slot_id = -1;
    int32_t seq_id = -1;
    int32_t task_id = -1;
    int32_t slot_state = 0;
    int32_t n_decoded = 0;
    int32_t prompt_n_tokens = 0;
    int32_t task_n_tokens = 0;
    int32_t i_batch = -1;
    bool is_generating = false;
    std::vector<int32_t> prompt_token_ids;
    std::string oracle_sample_id;
};

void rpp_live_configure(const common_params & params);

void rpp_live_attach_eval_callback(common_params & params);

int64_t rpp_live_pre_decode(
        const llama_batch & batch,
        const std::vector<rpp_live_slot_info> & slots,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch);

void rpp_live_post_decode(
        int64_t decode_call_id,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int32_t runtime_n_ubatch,
        int32_t decode_ret);

void rpp_live_pre_batch(
        const std::vector<rpp_live_slot_info> & slots,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch);
