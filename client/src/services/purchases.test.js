import {
  connect,
  disconnect,
  fetchUnlockProduct,
  buyUnlock,
  completePurchase,
  addPurchaseUpdatedListener,
  addPurchaseErrorListener,
  UNLOCK_SKU,
} from './purchases';
import * as RNIap from 'react-native-iap';

jest.mock('react-native-iap');

describe('purchases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('connect/disconnect delegate straight to the SDK', () => {
    connect();
    expect(RNIap.initConnection).toHaveBeenCalled();

    disconnect();
    expect(RNIap.endConnection).toHaveBeenCalled();
  });

  it('fetchUnlockProduct requests the unlock SKU and returns the first match', async () => {
    const product = { productId: UNLOCK_SKU, localizedPrice: '£2.99' };
    RNIap.getProducts.mockResolvedValue([product]);

    const result = await fetchUnlockProduct();

    expect(RNIap.getProducts).toHaveBeenCalledWith({ skus: [UNLOCK_SKU] });
    expect(result).toEqual(product);
  });

  it('fetchUnlockProduct returns null when the store has nothing for that SKU', async () => {
    RNIap.getProducts.mockResolvedValue([]);
    expect(await fetchUnlockProduct()).toBeNull();
  });

  it('buyUnlock requests a purchase for the unlock SKU', () => {
    buyUnlock();
    expect(RNIap.requestPurchase).toHaveBeenCalledWith({ sku: UNLOCK_SKU });
  });

  it('completePurchase finishes the transaction as non-consumable', () => {
    const purchase = { productId: UNLOCK_SKU, transactionId: 'abc' };
    completePurchase(purchase);
    expect(RNIap.finishTransaction).toHaveBeenCalledWith({ purchase, isConsumable: false });
  });

  it('wires listeners straight to the SDK', () => {
    const onUpdate = () => {};
    const onError = () => {};

    addPurchaseUpdatedListener(onUpdate);
    expect(RNIap.purchaseUpdatedListener).toHaveBeenCalledWith(onUpdate);

    addPurchaseErrorListener(onError);
    expect(RNIap.purchaseErrorListener).toHaveBeenCalledWith(onError);
  });
});
