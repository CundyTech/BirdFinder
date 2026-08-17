import {
  initConnection,
  endConnection,
  getProducts,
  requestPurchase,
  finishTransaction,
  purchaseUpdatedListener,
  purchaseErrorListener,
} from 'react-native-iap';

// TODO: create this as a non-consumable/managed product in Play Console
// (and the App Store equivalent) before release.
export const UNLOCK_SKU = 'com.cundytech.birdfinder.unlock_forever';

export function connect() {
  return initConnection();
}

export function disconnect() {
  return endConnection();
}

export async function fetchUnlockProduct() {
  const products = await getProducts({ skus: [UNLOCK_SKU] });
  return products?.[0] ?? null;
}

export function buyUnlock() {
  return requestPurchase({ sku: UNLOCK_SKU });
}

// isConsumable is false — this is a one-time unlock, not something that can
// be bought again after being used up.
export function completePurchase(purchase) {
  return finishTransaction({ purchase, isConsumable: false });
}

export function addPurchaseUpdatedListener(listener) {
  return purchaseUpdatedListener(listener);
}

export function addPurchaseErrorListener(listener) {
  return purchaseErrorListener(listener);
}
