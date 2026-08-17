import { useCallback, useEffect, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { useRewardedAd, TestIds } from 'react-native-google-mobile-ads';
import { grantAdReward } from '../store/filmSlice';

// TODO: swap for the real AdMob rewarded ad unit ID before release.
const AD_UNIT_ID = TestIds.REWARDED;

// Wraps the SDK's useRewardedAd hook: keeps an ad preloaded, and dispatches
// grantAdReward once the user has actually earned it (not just closed the ad).
export function useFilmRewardedAd() {
  const dispatch = useDispatch();
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
        dispatch(grantAdReward());
      }
      earnedThisShowRef.current = false;
      load();
    }
  }, [isClosed, dispatch, load]);

  const showAd = useCallback(() => {
    if (isLoaded) {
      show();
    }
  }, [isLoaded, show]);

  return { isLoaded, showAd, error };
}
