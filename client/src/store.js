import { configureStore } from '@reduxjs/toolkit';
import { api } from './services/api';
import { birdInfoApi } from './services/birdInfoApi';
import { setupListeners } from '@reduxjs/toolkit/query';

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    [birdInfoApi.reducerPath]: birdInfoApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware, birdInfoApi.middleware),
});

setupListeners(store.dispatch);
