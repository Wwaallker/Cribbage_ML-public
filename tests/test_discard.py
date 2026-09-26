from cribbage import CribbageEnv

env = CribbageEnv(opponent="smart")
env.reset()

# Ranks 0,1,2,3 (low run cards) + a pair of 6's in two suits.
# Hand-verified: discarding the pair [6,19] scores 6 for the DEALER (run of 4 kept + pair fed to crib)
# but only 2 for the NON-DEALER (pair fed to the AI's crib is bad) -- and every other discard
# option tops out at 4, so the two roles should give genuinely different, unambiguous answers.
env.comp_hand = [0, 14, 28, 3, 6, 19]

env.is_ai_dealer = False   # comp_is_dealer = True
discard_as_dealer = env._comp_discard()
print("As dealer, discarded:", sorted(discard_as_dealer))
# Expected: [6, 19] -- uniquely best (6 points) by feeding its own crib the pair

env.is_ai_dealer = True    # comp_is_dealer = False
discard_as_nondealer = env._comp_discard()
print("As non-dealer, discarded:", sorted(discard_as_nondealer))
# Expected: NOT [6, 19] -- should be one of [14,28], [14,3], or [3,28] instead (all score 4,
# a three-way tie, but all strictly better than feeding the AI's crib that pair for only 2)