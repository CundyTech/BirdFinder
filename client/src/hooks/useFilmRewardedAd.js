import { useCallback, useEffect, useRef } from 'react';
import { useRewardedAd, TestIds } from 'react-native-google-mobile-ads';

// TODO: swap for the real AdMob rewarded ad unit ID before release.
const AD_UNIT_ID = TestIds.REWARDED;

// Wraps the SDK's useRewardedAd hook: keeps an ad preloaded, and calls
// onEarned once the user has actually earned the reward (not just closed
// the ad) — what that reward actually grants is the caller's call, since
// different flows redeem the same rewarded ad differently (OutOfFilmModal
// tops up the Film balance; FilmChoiceModal covers one identification for
// free without touching it at all).
export function useFilmRewardedAd(onEarned) {
  const { isLoaded, isClosed, isEarnedReward, load, show, error } = useRewardedAd(AD_UNIT_ID, {
    requestNonPersonalizedAdsOnly: true,
  });
  const earnedThisShowRef = useRef(false);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (isEarnedReward) {
      earnedThisShowRef.current = true;
    }
  }, [isEarnedReward]);

  useEffect(() => {
    if (isClosed) {
      if (earnedThisShowRef.current) {
        onEarned();
      }
      earnedThisShowRef.current = false;
      load();
    }
  }, [isClosed, onEarned, load]);

  const showAd = useCallback(() => {
    if (isLoaded) {
      show();
    }
  }, [isLoaded, show]);

  return { isLoaded, showAd, error };
}
