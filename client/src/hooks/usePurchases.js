import { useCallback, useEffect, useRef, useState } from 'react';
import { useDispatch } from 'react-redux';
import {
  connect,
  disconnect,
  fetchUnlockProduct,
  buyUnlock,
  completePurchase,
  addPurchaseUpdatedListener,
  addPurchaseErrorListener,
  UNLOCK_SKU,
} from '../services/purchases';
import { unlockForever } from '../store/premiumSlice';

// There's no backend to validate receipts against, so a purchase is trusted
// as soon as the store reports it as completed. Fine for a small hobby app;
// would need server-side verification before this could be worth spoofing.
export function usePurchases() {
  const dispatch = useDispatch();
  const [product, setProduct] = useState(null);
  const [purchasing, setPurchasing] = useState(false);
  const [error, setError] = useState(null);
  const connectedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    connect()
      .then(() => {
        connectedRef.current = true;
        return fetchUnlockProduct();
      })
      .then((fetched) => {
        if (!cancelled) setProduct(fetched);
      })
      .catch((err) => {
        if (!cancelled) setError(err);
      });

    const updateSub = addPurchaseUpdatedListener(async (purchase) => {
      if (purchase.productId !== UNLOCK_SKU) return;
      try {
        await completePurchase(purchase);
        await dispatch(unlockForever());
      } catch (err) {
        setError(err);
      } finally {
        setPurchasing(false);
      }
    });

    const errorSub = addPurchaseErrorListener((err) => {
      setError(err);
      setPurchasing(false);
    });

    return () => {
      cancelled = true;
      updateSub.remove();
      errorSub.remove();
      if (connectedRef.current) {
        disconnect();
        connectedRef.current = false;
      }
    };
  }, [dispatch]);

  const purchaseUnlock = useCallback(() => {
    setError(null);
    setPurchasing(true);
    buyUnlock().catch((err) => {
      setError(err);
      setPurchasing(false);
    });
  }, []);

  return { product, purchasing, error, purchaseUnlock };
}
