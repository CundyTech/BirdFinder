import { configureStore } from '@reduxjs/toolkit';
import { api } from '../services/api';
import { birdInfoApi } from '../services/birdInfoApi';
import lifeListReducer from './lifeListSlice';
import filmReducer from './filmSlice';
import { setupListeners } from '@reduxjs/toolkit/query';

export const store = configureStore({
  reducer: {
    [api.reducerPath]: api.reducer,
    [birdInfoApi.reducerPath]: birdInfoApi.reducer,
    lifeList: lifeListReducer,
    film: filmReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware, birdInfoApi.middleware),
});

setupListeners(store.dispatch);
