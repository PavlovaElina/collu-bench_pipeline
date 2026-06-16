#lang racket
(define (minPath grid k)
  (define n (length grid))
  (define (cell i j) (list-ref (list-ref grid i) j))
  (define pos (for*/first ([i (in-range n)] [j (in-range n)] #:when (= (cell i j) 1)) (cons i j)))
  (define i0 (car pos)) (define j0 (cdr pos))
  (define neighbors
    (for/list ([d '((-1 . 0) (1 . 0) (0 . -1) (0 . 1))]
               #:when (let ([ni (+ i0 (car d))] [nj (+ j0 (cdr d))])
                        (and (>= ni 0) (< ni n) (>= nj 0) (< nj n))))
      (cell (+ i0 (car d)) (+ j0 (cdr d)))))
  (define m (apply min neighbors))
  (for/list ([idx (in-range k)]) (if (even? idx) 1 m)))
