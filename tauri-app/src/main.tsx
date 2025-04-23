import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

import {
  createBrowserRouter,
  RouterProvider,
} from "react-router";
import Slide1_1 from "./slides/Slide1_1";
import Slide1_2 from "./slides/Slide1_2";
import Slide2 from "./slides/Slide2";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "/slides/Slide1_1",
    element: <Slide1_1 />,
  },
  {
    path: "/slides/Slide1_2",
    element: <Slide1_2 />,
  },
  {
    path: "/slides/Slide2",
    element: <Slide2 />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
