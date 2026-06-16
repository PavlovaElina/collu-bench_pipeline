#lang racket
(define (iscube a)
  (define b (abs a))
  (let loop ([i 0])
    (cond [(> (* i i i) b) #f]
          [(= (* i i i) b) #t]
          [else (loop (+ i 1))])))
