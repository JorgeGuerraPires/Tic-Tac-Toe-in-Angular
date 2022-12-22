import { Component, OnInit } from '@angular/core';

//TensorFlow.js
import * as tf from '@tensorflow/tfjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'app';


  ngOnInit() {
    this.game();
  }

  game() {
    tf.ready().then(() => {
      const modelPath = './assets/model/ttt_model.json'

      tf.tidy(() => {
        tf.loadLayersModel(modelPath).then((model) => {
          // Three board states
          const emptyBoard = tf.zeros([9])
          const betterBlockMe = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1])

          const betterBlockMe2 = tf.tensor([-1, 0, 1, 0, -1, 0, 0, 0, 0])

          const goForTheKill = tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1])


          // Stack states into a shape [3, 9]
          const matches = tf.stack([emptyBoard, betterBlockMe2, goForTheKill])
          const result = model.predict(matches) as tf.Tensor

          // Log the results
          result.reshape([3, 3, 3]).print()

        })
      })
    })
  }
}
