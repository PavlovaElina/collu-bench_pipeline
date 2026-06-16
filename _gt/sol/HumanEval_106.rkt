#lang racket
(define (f n)
  (for/list ([i (in-range 1 (+ n 1))])
    (if (even? i)
        (for/product ([k (in-range 1 (+ i 1))]) k)
        (for/sum ([k (in-range 1 (+ i 1))]) k))))
