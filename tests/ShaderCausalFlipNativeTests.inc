// Actual VideoOut/PM4 paths under child-process opt-in. No duplicate Flip predicate.
namespace CausalFlipTest {
namespace CD = Libs::Graphics::CausalDiagnostics;

static CD::FamilySnapshot ReadFamily() {
  CD::Snapshot snapshot{};
  Require("CausalFlip", "snapshot", CD::RuntimeRecorder().ReadSnapshot(snapshot),
          "diagnostic recorder is unavailable");
  auto family = snapshot.family[static_cast<size_t>(CD::Family::Flip)];
  Require("CausalFlip", "coherent family", family.valid,
          "Flip family was contended after the fixture threads closed");
  return family;
}

static uint64_t LastSequence(const CD::FamilySnapshot &family) {
  uint64_t sequence = 0;
  for (size_t i = 0; i < family.count; ++i)
    sequence = std::max(sequence, family.records[i].sequence);
  return sequence;
}

static void CheckTransitions(const CD::FamilySnapshot &before,
                             const CD::FamilySnapshot &after,
                             bool gpu_source, size_t expected_requests,
                             bool require_rate_refusal) {
  const auto first_sequence = LastSequence(before);
  Require("CausalFlip", "retention", after.dropped == before.dropped,
          "the bounded fixture dropped a Flip record; its causal oracle is incomplete");
  std::vector<const CD::Record *> records;
  for (size_t i = 0; i < after.count; ++i)
    if (after.records[i].sequence > first_sequence)
      records.push_back(&after.records[i]);

  size_t requests = 0;
  bool no_owner = false;
  bool rate_refusal = false;
  for (const auto *record : records) {
    if (record->event == CD::Event::FlipNoOwner) {
      no_owner = true;
      Require("CausalFlip", "no owner state",
              (record->data[0] == 0) == (record->id == 0) &&
                  (record->data[1] == 0) == (record->related == 0) &&
                  (record->data[0] != 0 || (record->flags & 15u) == 15u) &&
                  (record->data[1] != 0 || (record->flags >> 4u) == 15u),
              "actual no-owner return lost queue counts or explicit absent front states");
    }
    if (record->event != CD::Event::FlipReserved) continue;
    ++requests;
    Require("CausalFlip", "request identity",
            record->id != 0 && record->related == record->id &&
                record->data[0] != 0 && record->data[1] != 0 &&
                static_cast<uint32_t>(record->data[2]) == UINT32_MAX &&
                ((record->data[2] >> 40u) & 1u) == static_cast<uint64_t>(gpu_source),
            "the real blank-buffer request lost source/group/port/generation identity");

    const auto find = [&](CD::Event event) -> const CD::Record * {
      const auto found = std::find_if(records.begin(), records.end(), [&](const auto *r) {
        return r->event == event && r->domain == record->domain && r->id == record->id;
      });
      return found == records.end() ? nullptr : *found;
    };
    const auto *prepared = find(CD::Event::FlipPrepared);
    const auto *ready = find(CD::Event::FlipReady);
    const auto *dispatch = find(CD::Event::FlipDispatch);
    const auto *complete = find(CD::Event::FlipGuestComplete);
    Require("CausalFlip", "real transition chain",
            prepared && ready && dispatch && complete &&
                record->sequence < prepared->sequence && prepared->sequence < ready->sequence &&
                ready->sequence < dispatch->sequence && dispatch->sequence < complete->sequence,
            "a real completed flip is missing its ordered reserve/prepare/ready/dispatch/guest-complete events");
    Require("CausalFlip", "request continuity",
            prepared->data[0] == record->data[0] && ready->data[0] == record->data[0] &&
                dispatch->data[0] == record->data[0] && complete->data[0] == record->data[0] &&
                prepared->data[1] == record->data[1] && ready->data[1] == record->data[1] &&
                dispatch->data[1] == record->data[1] && complete->data[1] == record->data[1] &&
                ((prepared->data[2] >> 41u) & 1u) != 0 &&
                ((ready->data[2] >> 32u) & 255u) == 1u &&
                ((dispatch->data[2] >> 32u) & 255u) == 2u,
            "observed frame readiness or generation changed while the real request completed");
    Require("CausalFlip", "cross-thread ancestry",
            dispatch->parent_sequence == ready->sequence &&
                dispatch->parent_domain == ready->domain && dispatch->parent_id == ready->id &&
                dispatch->parent_family == CD::Family::Flip &&
                complete->parent_sequence == dispatch->sequence &&
                complete->parent_domain == dispatch->domain && complete->parent_id == dispatch->id &&
                complete->parent_family == CD::Family::Flip,
            "the present thread lost the ready callback or dispatch parent");
    for (size_t guest = 3; guest < 6; ++guest)
      Require("CausalFlip", "guest origin",
              dispatch->data[guest] == ready->data[guest] &&
                  complete->data[guest] == ready->data[guest],
              "the ready callback's full guest tuple was not retained through completion");

    for (const auto *detail : records) {
      if (detail->event != CD::Event::FlipNotDue || detail->domain != record->domain ||
          detail->id != record->id || (detail->flags & 0x80u) == 0) continue;
      const auto metadata = std::find_if(records.begin(), records.end(), [&](const auto *r) {
        return r->event == CD::Event::FlipNotDue && r->domain == detail->domain &&
               r->id == detail->id && r->timestamp_ns == detail->timestamp_ns &&
               r->flags == (detail->flags & 0x7fu);
      });
      Require("CausalFlip", "refusal pair", metadata != records.end(),
              "not-due current values lost their captured-request metadata");
      if ((detail->flags & 3u) == 2u) {
        rate_refusal = true;
        Require("CausalFlip", "actual rate return",
                ((detail->flags >> 2u) & 3u) == 1u &&
                    detail->data[0] == (*metadata)->data[1] &&
                    detail->data[1] - detail->data[2] < 2u &&
                    detail->sequence < dispatch->sequence,
                "actual rate-1 refusal did not preserve the checked generation/vblank values");
      }
    }
  }
  Require("CausalFlip", "completed request count", requests == expected_requests,
          "the real fixture's request count differs from the retained diagnostic chain");
  Require("CausalFlip", "actual no-owner return", no_owner,
          "the real fixture never recorded its idle no-owner return");
  Require("CausalFlip", "actual rate refusal", !require_rate_refusal || rate_refusal,
          "the real FlipRate fixture never observed a ready group refused by its rate");
}
} // namespace CausalFlipTest

void CheckCausalFlipDiagnostics() {
  namespace CD = Libs::Graphics::CausalDiagnostics;
  Require("CausalFlip", "child-process opt-in", CD::Enabled(),
          "run with KYTY_CAUSAL_DIAG=1 before process startup");
  const auto before_pm4 = CausalFlipTest::ReadFamily();
  CheckPm4Flip();
  const auto after_pm4 = CausalFlipTest::ReadFamily();
  CausalFlipTest::CheckTransitions(before_pm4, after_pm4, true, 1, false);
  CheckFlipRate();
  const auto after_rate = CausalFlipTest::ReadFamily();
  CausalFlipTest::CheckTransitions(after_pm4, after_rate, false, 4, true);
  std::printf("[host]    %-32s ok\n", "CausalFlipProductionPaths");
}
